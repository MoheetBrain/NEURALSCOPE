/**
 * Small in-memory LRU for view responses, keyed by (endpoint, run, params).
 * View responses for a stored run are deterministic, so they can be cached for the
 * session; anything that can change (run detail, lists) carries a version in its key.
 */
export class LruCache<V> {
  private map = new Map<string, V>();

  constructor(private readonly max = 150) {}

  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      // refresh recency
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }

  peek(key: string): V | undefined {
    return this.map.get(key);
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  set(key: string, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.max) {
      const oldest = this.map.keys().next().value;
      if (oldest === undefined) break;
      this.map.delete(oldest);
    }
  }

  delete(key: string): void {
    this.map.delete(key);
  }

  /** Remove every key starting with `prefix`. */
  invalidatePrefix(prefix: string): void {
    for (const k of [...this.map.keys()]) if (k.startsWith(prefix)) this.map.delete(k);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}

/** Shared cache for all workspace view fetches. */
export const viewCache = new LruCache<unknown>(160);
