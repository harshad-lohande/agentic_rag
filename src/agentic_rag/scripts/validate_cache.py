# src/agentic_rag/scripts/validate_cache.py

import redis
import weaviate
from rich.console import Console
from rich.table import Table

from agentic_rag.config import settings
from agentic_rag.logging_config import setup_logging

# Initialize rich console for better output
console = Console()


def validate_cache_integrity():
    """
    Validates the integrity of the semantic cache by performing a comprehensive
    three-way comparison between the Weaviate store, the Redis index, and
    the actual Redis cache entries.
    """
    setup_logging()
    console.print(
        "[bold cyan]--- Starting Comprehensive Cache Integrity Validation ---[/bold cyan]"
    )

    try:
        # Connect to Weaviate
        weaviate_client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST, port=settings.WEAVIATE_PORT
        )
        cache_collection = weaviate_client.collections.get(
            settings.SEMANTIC_CACHE_INDEX_NAME
        )
        console.print("✅ [green]Successfully connected to Weaviate.[/green]")

        # Connect to Redis
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True,
        )
        redis_client.ping()
        console.print("✅ [green]Successfully connected to Redis.[/green]")

        # 1. Get all UUIDs from Weaviate
        weaviate_uuids = set()
        for item in cache_collection.iterator(
            include_vector=False, return_properties=["cache_id"]
        ):
            cache_id = item.properties.get("cache_id")
            if cache_id:
                weaviate_uuids.add(str(cache_id))
        console.print(f"\nFound {len(weaviate_uuids)} vector entries in Weaviate.")

        # 2. Get all UUIDs from the Redis ZSET index
        redis_index_uuids = set(
            str(x) for x in redis_client.zrange("cache_index", 0, -1)
        )
        console.print(f"Found {len(redis_index_uuids)} entries in Redis 'cache_index'.")

        # 3. Check which indexed UUIDs have actual data in Redis
        stale_index_entries = set()
        active_redis_entries = set()
        indexed_ids_list = list(redis_index_uuids)

        if indexed_ids_list:
            # Use a pipeline for efficient checking
            pipe = redis_client.pipeline()
            for cache_id in indexed_ids_list:
                pipe.exists(f"cache_entry:{cache_id}")

            exists_results = pipe.execute()

            for i, exists in enumerate(exists_results):
                if exists:
                    active_redis_entries.add(indexed_ids_list[i])
                else:
                    stale_index_entries.add(indexed_ids_list[i])

        console.print(
            f"Found {len(active_redis_entries)} active data entries in Redis ('cache_entry:*')."
        )

        console.print("\n[bold cyan]--- Validation Report ---[/bold cyan]")

        # 4. Find inconsistencies
        weaviate_orphans = weaviate_uuids - redis_index_uuids
        redis_index_orphans = redis_index_uuids - weaviate_uuids

        is_perfectly_synced = (
            not weaviate_orphans and not redis_index_orphans and not stale_index_entries
        )

        if is_perfectly_synced:
            console.print(
                "✅ [bold green]Cache is perfectly in sync. No inconsistencies found.[/bold green]"
            )
            return

        # Report Stale Redis Index Entries (Problem from logs)
        if stale_index_entries:
            table = Table(
                title="🚨 Stale Redis Index Entries (Causes incorrect stats) 🚨"
            )
            table.add_column("Stale UUID in 'cache_index'", style="red", no_wrap=True)
            for uuid in stale_index_entries:
                table.add_row(str(uuid))
            console.print(table)
            console.print(
                "[red]These IDs exist in the Redis index, but their data has expired. "
                "This is why '/cache/stats' is incorrect. The background GC will clean these up.[/red]\n"
            )
        else:
            console.print(
                "✅ [green]Redis 'cache_index' is clean. No stale entries found.[/green]"
            )

        # Report Weaviate orphans (in Weaviate but not Redis)
        if weaviate_orphans:
            table = Table(title="⚠️ Weaviate Orphan Entries (Will not expire) ⚠️")
            table.add_column("Orphan UUID in Weaviate", style="yellow", no_wrap=True)
            for uuid in weaviate_orphans:
                table.add_row(str(uuid))
            console.print(table)
            console.print(
                "[yellow]These entries exist in Weaviate but are not tracked in the Redis index. "
                "The background GC will clean these up.[/yellow]\n"
            )
        else:
            console.print(
                "✅ [green]Weaviate is clean. No orphan vector entries found.[/green]"
            )

        # Report Redis index orphans (in Redis index but not Weaviate)
        if redis_index_orphans:
            table = Table(title="⚠️ Redis Index Orphan Entries (Harmless clutter) ⚠️")
            table.add_column(
                "Orphan UUID in Redis Index", style="magenta", no_wrap=True
            )
            for uuid in redis_index_orphans:
                table.add_row(str(uuid))
            console.print(table)
            console.print(
                "[magenta]These IDs exist in the Redis index but have no matching vector in Weaviate. "
                "The background GC should clean these up.[/magenta]\n"
            )
        else:
            console.print(
                "✅ [green]Redis 'cache_index' is consistent with Weaviate.[/green]"
            )

    except Exception as e:
        console.print(
            f"❌ [bold red]An error occurred during validation: {e}[/bold red]"
        )
    finally:
        if "weaviate_client" in locals():
            weaviate_client.close()
        console.print("\n[bold cyan]--- Validation Complete ---[/bold cyan]")


if __name__ == "__main__":
    validate_cache_integrity()
