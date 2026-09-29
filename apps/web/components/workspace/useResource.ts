"use client";

import { useEffect, useEffectEvent, useState } from "react";
import { isAbortError } from "@/lib/api/client";
import { viewCache } from "@/lib/api/cache";

export interface Resource<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  /** `data` belongs to a previous key (kept while the new one loads). */
  stale: boolean;
  reload: () => void;
}

export interface ResourceOptions {
  /** Delay before fetching (coalesces rapid key changes, e.g. arrow-key navigation). */
  debounceMs?: number;
  /** Keep showing the previous key's data (dimmed) while loading. Default true. */
  keepPrevious?: boolean;
  /** Use the shared in-memory view cache. Default true. */
  cache?: boolean;
}

interface State<T> {
  key: string | null;
  data?: T;
  error?: unknown;
  nonce: number;
}

/**
 * Fetch keyed data with caching and cancellation. A new key aborts the in-flight
 * request of the previous key (AbortController); `key === null` means "nothing to load".
 * Loading is derived (state.key !== key) so no state is set synchronously in effects.
 */
export function useResource<T>(
  key: string | null,
  fetcher: (signal: AbortSignal) => Promise<T>,
  { debounceMs = 0, keepPrevious = true, cache = true }: ResourceOptions = {},
): Resource<T> {
  const [state, setState] = useState<State<T>>({ key: null, nonce: 0 });
  const [nonce, setNonce] = useState(0);
  const load = useEffectEvent((signal: AbortSignal) => fetcher(signal));

  useEffect(() => {
    if (key === null) return;
    const ctl = new AbortController();
    const cached = cache ? (viewCache.get(key) as T | undefined) : undefined;
    if (cached !== undefined) {
      queueMicrotask(() => {
        if (!ctl.signal.aborted) setState({ key, data: cached, nonce });
      });
      return () => ctl.abort();
    }
    const timer = setTimeout(() => {
      load(ctl.signal).then(
        (d) => {
          if (ctl.signal.aborted) return;
          if (cache) viewCache.set(key, d);
          setState({ key, data: d, nonce });
        },
        (e) => {
          if (ctl.signal.aborted || isAbortError(e)) return;
          setState({ key, error: e, nonce });
        },
      );
    }, debounceMs);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
  }, [key, nonce, debounceMs, cache]);

  const reload = () => {
    if (key !== null) viewCache.delete(key);
    setNonce((n) => n + 1);
  };

  if (key === null) return { data: undefined, error: undefined, loading: false, stale: false, reload };
  if (state.key === key && state.nonce === nonce) {
    return { data: state.data, error: state.error, loading: false, stale: false, reload };
  }
  const cached = cache ? (viewCache.peek(key) as T | undefined) : undefined;
  if (cached !== undefined) return { data: cached, error: undefined, loading: false, stale: false, reload };
  const prev = keepPrevious && state.error === undefined ? state.data : undefined;
  return { data: prev, error: undefined, loading: true, stale: prev !== undefined, reload };
}

/** Ticking clock for elapsed timers / relative times (0 until mounted). */
export function useNow(intervalMs = 1000, active = true): number {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!active) return;
    const tick = () => setNow(Date.now());
    const first = setTimeout(tick, 0);
    const id = setInterval(tick, intervalMs);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, [intervalMs, active]);
  return now;
}
