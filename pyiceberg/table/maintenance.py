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
from __future__ import annotations

import itertools
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyiceberg.expressions import BooleanExpression
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.utils.properties import property_as_int

if TYPE_CHECKING:
    import pyarrow as pa

    from pyiceberg.io import FileIO
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import FileScanTask, Table
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.update.snapshot import ExpireSnapshots
    from pyiceberg.typedef import Record

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of a single data file rewrite group.

    Attributes:
        data_files: The replacement data files. Empty when every input record was removed by deletes.
        rewritten_data_files: The input data files that were replaced.
    """

    data_files: tuple[DataFile, ...]
    rewritten_data_files: tuple[DataFile, ...]


class _RewriteDataFilesWriter:
    """Writes compacted data files for one (partition spec, partition) group.

    The output files keep the input group's partition spec and partition values, so files written
    under historical partition specs are rewritten under that same spec instead of being silently
    moved to the current default spec.
    """

    def __init__(
        self,
        table_metadata: TableMetadata,
        io: FileIO,
        spec: PartitionSpec,
        partition: Record,
        write_uuid: uuid.UUID,
        target_file_size_bytes: int,
    ) -> None:
        self._table_metadata = table_metadata
        self._io = io
        self._spec = spec
        self._partition = partition
        self._write_uuid = write_uuid
        self._target_file_size_bytes = target_file_size_bytes
        self._counter = itertools.count(0)

    def write(self, arrow_table: pa.Table) -> tuple[DataFile, ...]:
        import pyarrow as pa

        from pyiceberg.io.fileformat import FileFormatFactory
        from pyiceberg.io.pyarrow import _to_requested_schema
        from pyiceberg.partitioning import PartitionFieldValue, PartitionKey
        from pyiceberg.table import DOWNCAST_NS_TIMESTAMP_TO_US_ON_WRITE, TableProperties
        from pyiceberg.table.locations import load_location_provider
        from pyiceberg.utils.config import Config

        if arrow_table.num_rows == 0:
            return ()

        downcast_ns_timestamp_to_us = Config().get_bool(DOWNCAST_NS_TIMESTAMP_TO_US_ON_WRITE) or False
        table_schema = self._table_metadata.schema()
        output_format = FileFormat(
            self._table_metadata.properties.get(TableProperties.WRITE_FILE_FORMAT, TableProperties.WRITE_FILE_FORMAT_DEFAULT)
        )
        format_model = FileFormatFactory.get(output_format)
        location_provider = load_location_provider(
            table_location=self._table_metadata.location, table_properties=self._table_metadata.properties
        )
        partition_key = PartitionKey(
            field_values=[
                PartitionFieldValue(field=field, value=self._partition[index]) for index, field in enumerate(self._spec.fields)
            ],
            partition_spec=self._spec,
            schema=table_schema,
        )

        # Attach Iceberg field ids to the Arrow schema so the written file carries them, matching
        # the regular write path.
        batches = [
            _to_requested_schema(
                requested_schema=table_schema,
                file_schema=table_schema,
                batch=batch,
                downcast_ns_timestamp_to_us=downcast_ns_timestamp_to_us,
                include_field_ids=True,
                format_model=format_model,
            )
            for batch in arrow_table.to_batches()
        ]
        arrow_table = pa.Table.from_batches(batches)

        # Bin-pack on uncompressed Arrow bytes, matching the regular write path: accumulate
        # batches until the target weight is reached rather than splitting by row count, so a
        # group of small files collapses into a single output file below the target size.
        from pyiceberg.utils.bin_packing import PackingIterator

        batches = list(arrow_table.to_batches())
        packed_groups = PackingIterator(
            items=batches,
            target_weight=self._target_file_size_bytes,
            lookback=len(batches),
            weight_func=lambda batch: batch.nbytes,
            largest_bin_first=False,
        )

        data_files: list[DataFile] = []
        for packed_batches in packed_groups:
            chunk_table = pa.Table.from_batches(packed_batches, schema=arrow_table.schema)
            file_name = f"00000-{next(self._counter)}-{self._write_uuid}.{format_model.file_extension()}"
            file_path = location_provider.new_data_location(file_name, partition_key)
            output_file = self._io.new_output(file_path)

            writer = format_model.create_writer(output_file, table_schema, self._table_metadata.properties)
            with writer:
                writer.write(chunk_table)
            statistics = writer.result()

            data_files.append(
                DataFile.from_args(
                    content=DataFileContent.DATA,
                    file_path=file_path,
                    file_format=output_format,
                    partition=self._partition,
                    file_size_in_bytes=len(output_file),
                    sort_order_id=None,
                    spec_id=self._spec.spec_id,
                    equality_ids=None,
                    key_metadata=None,
                    **statistics.to_serialized_dict(),
                )
            )

        return tuple(data_files)


class MaintenanceTable:
    tbl: Table

    def __init__(self, tbl: Table) -> None:
        self.tbl = tbl

    def expire_snapshots(self) -> ExpireSnapshots:
        """Return an ExpireSnapshots builder for snapshot expiration operations.

        Returns:
            ExpireSnapshots builder for configuring and executing snapshot expiration.
        """
        from pyiceberg.table import Transaction
        from pyiceberg.table.update.snapshot import ExpireSnapshots

        return ExpireSnapshots(transaction=Transaction(self.tbl, autocommit=True))

    def rewrite_data_files(
        self,
        predicate: BooleanExpression | None = None,
        *,
        min_input_files: int = 5,
        target_file_size_bytes: int | None = None,
        case_sensitive: bool = True,
        branch: str | None = "main",
    ) -> list[RewriteResult]:
        """Compact small data files of the selected partitions into larger files.

        The rewrite reads every selected data file through the table scan, so all applicable
        position deletes are applied before the replacement files are written. Deleted records
        therefore stay deleted and are not resurrected by the compaction. Files written under
        historical partition specs are rewritten under their own spec, grouped per partition.

        Only the selected input files are replaced in a single REPLACE snapshot. Files appended
        by another writer while the compaction runs are untouched: the commit retries against the
        refreshed head and carries those entries over. The commit fails explicitly if an input
        file was concurrently deleted or received new delete files, instead of publishing stale
        output.

        Args:
            predicate: Optional filter selecting the data files to compact. When None, every live
                data file of the (branch) snapshot is eligible. Only files whose whole file matches
                the filter (inclusive metrics evaluation) are selected.
            min_input_files: Minimum number of files a (spec, partition) group must contain to be
                rewritten. Groups smaller than this are skipped.
            target_file_size_bytes: Approximate target size of the output files in uncompressed
                in-memory Arrow bytes. Defaults to the table's ``write.target-file-size-bytes``.
            case_sensitive: Whether the predicate's column names are case sensitive.
            branch: Branch to compact. Defaults to the main branch.

        Returns:
            One RewriteResult per rewritten group, in a stable order. Empty when nothing qualified.
        """
        from pyiceberg.expressions import AlwaysTrue
        from pyiceberg.expressions.visitors import ROWS_MIGHT_NOT_MATCH, _InclusiveMetricsEvaluator
        from pyiceberg.io.pyarrow import ArrowScan
        from pyiceberg.table import TableProperties, Transaction

        table = self.tbl
        table_metadata = table.metadata
        if branch is not None and table.snapshot_by_name(branch) is None:
            return []

        if target_file_size_bytes is None:
            target_file_size_bytes = property_as_int(
                properties=table_metadata.properties,
                property_name=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
                default=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT,
            )
        assert target_file_size_bytes is not None

        # Plan against the snapshot visible to this table instance. FileScanTask already resolves
        # the position delete files that apply to each data file, following the sequence number
        # rules of the table format.
        scan = table.scan(case_sensitive=case_sensitive)
        if branch is not None and branch != "main":
            scan = scan.use_ref(branch)
        if predicate is not None:
            scan = scan.filter(predicate)

        tasks = list(scan.plan_files())
        if predicate is not None:
            metrics_evaluator = _InclusiveMetricsEvaluator(table_metadata.schema(), predicate, case_sensitive=case_sensitive)
            tasks = [task for task in tasks if metrics_evaluator.eval(task.file) != ROWS_MIGHT_NOT_MATCH]

        groups = self._group_tasks(tasks)
        eligible_groups = [group for group in groups if len(group[2]) >= min_input_files]
        if not eligible_groups:
            logger.info("No file groups with at least %s input files found, skipping rewrite", min_input_files)
            return []

        write_uuid = uuid.uuid4()
        results: list[RewriteResult] = []

        transaction = Transaction(table, autocommit=True)
        with transaction.update_snapshot(branch=branch).rewrite_files(commit_uuid=write_uuid) as rewrite:
            arrow_scan = ArrowScan(
                table_metadata=transaction.table_metadata,
                io=table.io,
                projected_schema=transaction.table_metadata.schema(),
                row_filter=AlwaysTrue(),
                case_sensitive=case_sensitive,
            )
            for spec_id, partition, group_tasks in eligible_groups:
                spec = transaction.table_metadata.specs()[spec_id]
                input_files = tuple(task.file for task in group_tasks)
                # Reading via the scan tasks applies every applicable position delete, so the
                # compacted output only contains live records.
                arrow_table = arrow_scan.to_table(group_tasks)
                writer = _RewriteDataFilesWriter(
                    table_metadata=transaction.table_metadata,
                    io=table.io,
                    spec=spec,
                    partition=partition,
                    write_uuid=write_uuid,
                    target_file_size_bytes=target_file_size_bytes,
                )
                replacement_files = writer.write(arrow_table)

                for input_file in input_files:
                    rewrite.delete_data_file(input_file)
                for replacement_file in replacement_files:
                    rewrite.append_data_file(replacement_file)

                results.append(RewriteResult(data_files=replacement_files, rewritten_data_files=input_files))

        return results

    @staticmethod
    def _group_tasks(tasks: list[FileScanTask]) -> list[tuple[int, Record, list[FileScanTask]]]:
        """Group scan tasks by (partition spec id, partition record) in a deterministic order."""
        grouped: dict[tuple[int, Record], list[FileScanTask]] = defaultdict(list)
        for task in tasks:
            grouped[(task.file.spec_id, task.file.partition)].append(task)
        return [(spec_id, partition, grouped[(spec_id, partition)]) for spec_id, partition in sorted(grouped, key=repr)]
