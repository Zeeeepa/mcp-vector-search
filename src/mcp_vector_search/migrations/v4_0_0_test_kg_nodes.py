"""Migration v4.0.0: Add TestSuite, TestCase node tables and test relationship tables.

Issue #156. Adds:

- ``TestSuite`` node table
- ``TestCase`` node table
- ``TESTS`` (TestCase -> CodeEntity)
- ``BELONGS_TO_SUITE`` (TestCase -> TestSuite)
- ``USES_FIXTURE`` (TestCase -> CodeEntity)

Schema creation uses ``IF NOT EXISTS`` so this migration is idempotent and
safe to re-run on databases that already contain the new tables.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from ..core.knowledge_graph import KnowledgeGraph
from .migration import Migration, MigrationContext, MigrationResult, MigrationStatus

# Tables this migration creates; used for the "is needed" probe.
_NEW_TABLES: tuple[str, ...] = (
    "TestSuite",
    "TestCase",
    "TESTS",
    "BELONGS_TO_SUITE",
    "USES_FIXTURE",
)


def _kg_path(context: MigrationContext) -> Path:
    """Return the path where the KG database lives for this project."""
    return context.index_path / "knowledge_graph"


class TestKGNodesMigration(Migration):
    """Add TestSuite/TestCase nodes and TESTS/BELONGS_TO_SUITE/USES_FIXTURE edges."""

    version = "4.0.0"
    name = "test_kg_nodes"
    description = (
        "Add TestSuite, TestCase, TESTS, BELONGS_TO_SUITE, USES_FIXTURE "
        "to the knowledge graph"
    )

    def check_needed(self, context: MigrationContext) -> bool:
        """Migration is needed when the KG exists but lacks the new tables."""
        kg_root = _kg_path(context)
        # If there is no KG at all, there is nothing to migrate.
        if not kg_root.exists():
            return False

        try:
            kg = KnowledgeGraph(kg_root)
            kg.initialize_sync()
        except Exception as e:
            logger.debug(f"TestKGNodesMigration: failed to init KG: {e}")
            return False

        try:
            for table in _NEW_TABLES:
                # MATCH against unknown tables raises; if any check raises,
                # we treat the schema as out of date.
                try:
                    kg._execute_query(
                        f"MATCH (n:{table}) RETURN count(n) LIMIT 1"
                        if table in {"TestSuite", "TestCase"}
                        else f"MATCH ()-[r:{table}]->() RETURN count(r) LIMIT 1"
                    )
                except Exception:
                    return True
            return False
        finally:
            try:
                if kg.conn is not None:
                    # Best-effort connection cleanup; KuzuDB releases on GC.
                    kg.conn = None
                    kg.db = None
                    kg._initialized = False
            except Exception:
                pass

    def execute(self, context: MigrationContext) -> MigrationResult:
        """Recreate the schema (idempotent IF NOT EXISTS) on the KG."""
        if context.dry_run:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message="DRY RUN: Would add TestSuite/TestCase tables to KG",
            )

        kg_root = _kg_path(context)
        if not kg_root.exists():
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SKIPPED,
                message="No knowledge graph present — nothing to migrate",
                executed_at=datetime.now(),
            )

        try:
            kg = KnowledgeGraph(kg_root)
            kg.initialize_sync()
            # Calling _create_schema is idempotent (IF NOT EXISTS).
            kg._create_schema()
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message="Added TestSuite, TestCase, TESTS, BELONGS_TO_SUITE, USES_FIXTURE",
                executed_at=datetime.now(),
            )
        except Exception as e:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message=f"Failed to apply test KG schema: {e}",
                executed_at=datetime.now(),
            )

    def rollback(self, context: MigrationContext) -> bool:
        """Rollback not supported: KuzuDB tables added with IF NOT EXISTS."""
        logger.info(
            "Rollback not supported for v4.0.0_test_kg_nodes; "
            "drop tables manually if required."
        )
        return False
