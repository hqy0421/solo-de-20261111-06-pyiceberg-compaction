# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from pathlib import PosixPath

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyiceberg.expressions import EqualTo
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Operation
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType
from tests.catalog.test_base import InMemoryCatalog


def _schema() -> Schema:
    return Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )


def _partition_spec() -> PartitionSpec:
    return PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))


@pytest.fixture
def catalog(tmp_path: PosixPath) -> InMemoryCatalog:
    catalog = InMemoryCatalog("test.maintenance.catalog", warehouse=tmp_path.absolute().as_posix())
    catalog.create_namespace("default")
    return catalog


def _create_table(catalog: InMemoryCatalog, *, format_version: int = 2) -> Table:
    table = catalog.create_table("default.t", schema=_schema(), partition_spec=_partition_spec())
    if format_version == 2:
        table.transaction().upgrade_table_version(2)
    return table


def _append_file(table: Table, category: str, value: int) -> None:
    table.append(pa.table({"category": [category], "value": [value]}))


def _data_files(table: Table) -> list:
    return [task.file for task in table.scan().plan_files()]


def _add_position_delete(table: Table, data_file: DataFile, positions: tuple[int, ...] = (0,)) -> DataFile:
    delete_path = data_file.file_path.replace(".parquet", "-delete.parquet")
    delete_table = pa.table(
        {
            "file_path": pa.array([data_file.file_path] * len(positions), type=pa.large_string()),
            "pos": pa.array(positions, type=pa.int32()),
        }
    )
    with table.io.new_output(delete_path).create() as output_stream:
        pq.write_table(delete_table, output_stream)

    delete_file = DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path=delete_path,
        file_format=FileFormat.PARQUET,
        partition=data_file.partition,
        file_size_in_bytes=len(table.io.new_input(delete_path)),
        record_count=len(positions),
    )
    with table.transaction() as transaction:
        with transaction.update_snapshot().fast_append() as append:
            append.append_data_file(delete_file)
    return delete_file


def test_rewrite_compacts_small_files_into_one_per_partition(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(4):
        _append_file(table, "a", value)
    _append_file(table, "b", 10)

    results = table.maintenance.rewrite_data_files(min_input_files=2)

    assert len(results) == 1
    result = results[0]
    assert len(result.rewritten_data_files) == 4
    assert len(result.data_files) == 1
    assert {data_file.partition[0] for data_file in result.rewritten_data_files} == {"a"}

    compacted = catalog.load_table("default.t")
    assert len(_data_files(compacted)) == 2
    assert compacted.scan().to_arrow()["value"].to_pylist().count(10) == 1

    snapshot = compacted.current_snapshot()
    assert snapshot is not None
    assert snapshot.summary is not None
    assert snapshot.summary.operation == Operation.REPLACE
    assert snapshot.summary["added-data-files"] == "1"
    assert snapshot.summary["deleted-data-files"] == "4"
    assert snapshot.summary["total-records"] == "5"


def test_rewrite_skips_groups_below_min_input_files(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(2):
        _append_file(table, "a", value)

    assert table.maintenance.rewrite_data_files(min_input_files=5) == []
    assert len(_data_files(catalog.load_table("default.t"))) == 2


def test_rewrite_on_empty_table_returns_empty(catalog: InMemoryCatalog) -> None:
    table = catalog.create_table("default.empty", schema=_schema(), partition_spec=_partition_spec())

    assert table.maintenance.rewrite_data_files() == []


def test_rewrite_applies_position_deletes(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)

    target_file = next(file for file in _data_files(table) if file.record_count == 1)
    table = catalog.load_table("default.t")
    _add_position_delete(table, target_file)
    assert catalog.load_table("default.t").scan().to_arrow().num_rows == 2

    results = catalog.load_table("default.t").maintenance.rewrite_data_files(min_input_files=2)

    assert len(results) == 1
    compacted = catalog.load_table("default.t")
    assert len(_data_files(compacted)) == 1
    values = compacted.scan().to_arrow()["value"].to_pylist()
    assert len(values) == 2
    deleted_value = pa.parquet.read_table(target_file.file_path)["value"].to_pylist()[0]
    assert deleted_value not in values


def test_rewrite_with_all_rows_deleted_writes_no_replacement(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)
    for value in range(3):
        _append_file(table, "b", value + 10)

    table = catalog.load_table("default.t")
    for data_file in [file for file in _data_files(table) if file.partition[0] == "a"]:
        _add_position_delete(table, data_file)

    table = catalog.load_table("default.t")
    results = table.maintenance.rewrite_data_files(min_input_files=2)

    assert len(results) == 2
    empty_result = next(result for result in results if result.rewritten_data_files[0].partition[0] == "a")
    assert empty_result.data_files == ()

    compacted = catalog.load_table("default.t")
    remaining = _data_files(compacted)
    assert len(remaining) == 1
    assert remaining[0].partition[0] == "b"
    assert sorted(compacted.scan().to_arrow()["value"].to_pylist()) == [10, 11, 12]


def test_rewrite_preserves_historical_partition_specs(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)

    with table.transaction() as transaction:
        update_spec = transaction.update_spec()
        update_spec.add_identity("value")
        update_spec.commit()

    table = catalog.load_table("default.t")
    for _ in range(3):
        _append_file(table, "b", 100)

    table = catalog.load_table("default.t")
    specs = table.metadata.specs()
    input_spec_ids = {file.spec_id for file in _data_files(table)}
    assert input_spec_ids == {0, 1}

    results = table.maintenance.rewrite_data_files(min_input_files=2)

    assert len(results) == 2
    for result in results:
        input_spec_id = result.rewritten_data_files[0].spec_id
        assert {data_file.spec_id for data_file in result.data_files} == {input_spec_id}
        assert {data_file.partition for data_file in result.data_files} == {result.rewritten_data_files[0].partition}

    compacted = catalog.load_table("default.t")
    assert sorted(compacted.scan().to_arrow()["value"].to_pylist()) == [0, 1, 2, 100, 100, 100]
    assert {file.spec_id for file in _data_files(compacted)} == {0, 1}
    assert compacted.metadata.default_spec_id == specs[1].spec_id


def test_rewrite_predicate_selects_partition(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)
    for value in range(3):
        _append_file(table, "b", value + 10)

    results = table.maintenance.rewrite_data_files(EqualTo("category", "a"), min_input_files=2)

    assert len(results) == 1
    assert {data_file.partition[0] for data_file in results[0].rewritten_data_files} == {"a"}
    compacted = catalog.load_table("default.t")
    files = _data_files(compacted)
    # Partition a collapses 3 files into 1, partition b keeps its 3 untouched files.
    assert len(files) == 4
    assert sorted(compacted.scan().to_arrow()["value"].to_pylist()) == [0, 1, 2, 10, 11, 12]


def test_rewrite_preserves_concurrent_append(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)

    stale = catalog.load_table("default.t")
    catalog.load_table("default.t").append(pa.table({"category": ["a"], "value": [99]}))

    results = stale.maintenance.rewrite_data_files(min_input_files=2)

    assert len(results) == 1
    compacted = catalog.load_table("default.t")
    assert sorted(compacted.scan().to_arrow()["value"].to_pylist()) == [0, 1, 2, 99]
    files = _data_files(compacted)
    assert len(files) == 2
    assert compacted.current_snapshot().summary.operation == Operation.REPLACE


def test_rewrite_fails_when_input_file_concurrently_deleted(catalog: InMemoryCatalog) -> None:
    from pyiceberg.exceptions import ValidationException

    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)

    stale = catalog.load_table("default.t")
    other = catalog.load_table("default.t")
    target_file = _data_files(other)[0]
    with other.transaction() as transaction:
        with transaction.update_snapshot().overwrite() as overwrite:
            overwrite.delete_data_file(target_file)

    with pytest.raises(ValidationException, match="Data files were concurrently deleted"):
        stale.maintenance.rewrite_data_files(min_input_files=2)

    compacted = catalog.load_table("default.t")
    remaining_values = compacted.scan().to_arrow()["value"].to_pylist()
    assert len(remaining_values) == 2
    assert target_file.file_path not in {data_file.file_path for data_file in _data_files(compacted)}


def test_rewrite_fails_when_new_delete_applies_to_input_file(catalog: InMemoryCatalog) -> None:
    from pyiceberg.exceptions import ValidationException

    table = _create_table(catalog)
    for value in range(3):
        _append_file(table, "a", value)

    stale = catalog.load_table("default.t")
    other = catalog.load_table("default.t")
    target_file = _data_files(other)[0]
    _add_position_delete(other, target_file)

    with pytest.raises(ValidationException, match="found new delete for replace data file"):
        stale.maintenance.rewrite_data_files(min_input_files=2)

    compacted = catalog.load_table("default.t")
    assert compacted.scan().to_arrow().num_rows == 2


def test_rewrite_on_v1_table(catalog: InMemoryCatalog) -> None:
    table = _create_table(catalog, format_version=1)
    for value in range(3):
        _append_file(table, "a", value)
    for value in range(3):
        _append_file(table, "b", value + 10)

    results = table.maintenance.rewrite_data_files(min_input_files=2)

    assert len(results) == 2
    compacted = catalog.load_table("default.t")
    assert sorted(compacted.scan().to_arrow()["value"].to_pylist()) == [0, 1, 2, 10, 11, 12]
    snapshot = compacted.current_snapshot()
    assert snapshot.summary.operation == Operation.REPLACE
    assert snapshot.summary["added-data-files"] == "2"
    assert snapshot.summary["deleted-data-files"] == "6"
