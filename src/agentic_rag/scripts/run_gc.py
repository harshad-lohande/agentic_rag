# src/agentic_rag/scripts/run_gc.py

import asyncio
from agentic_rag.app.semantic_cache import get_semantic_cache
from agentic_rag.logging_config import setup_logging
from rich.console import Console

console = Console()


async def main():
    """
    Manually triggers the garbage collection process for the semantic cache
    to find and clean up orphan entries in Weaviate.
    """
    setup_logging()
    console.print("[bold cyan]--- Manual Cache Garbage Collection ---[/bold cyan]")

    try:
        cache = get_semantic_cache()

        # This will initialize the cache and its clients if not already done
        is_initialized = await cache._initialize_clients()

        if not is_initialized:
            console.print(
                "[bold red]❌ Failed to initialize semantic cache. Aborting GC.[/bold red]"
            )
            return

        console.print("✅ Cache initialized. Starting garbage collection process...")

        # The run_garbage_collection method now returns the number of cleaned entries
        cleaned_count = await cache.run_garbage_collection_manually()

        if cleaned_count > 0:
            console.print(
                f"[bold green]✅ Garbage collection complete. Cleaned up {cleaned_count} orphan entries.[/bold green]"
            )
        else:
            console.print(
                "[bold green]✅ Garbage collection complete. No orphan entries were found.[/bold green]"
            )

    except Exception as e:
        console.print(f"❌ [bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        # Gracefully shutdown the cache connections
        cache = get_semantic_cache()
        if cache._initialized:
            await cache.shutdown()
        console.print("\n[bold cyan]--- GC Process Finished ---[/bold cyan]")


if __name__ == "__main__":
    asyncio.run(main())
