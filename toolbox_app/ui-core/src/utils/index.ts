/**
 * Get the API base URL from environment or use empty string
 */
export function getApiBase(): string {
  return import.meta.env.VITE_API_BASE || '';
}

/**
 * Fetch helper with automatic API base prefixing
 */
export async function apiFetch(
  endpoint: string,
  options?: RequestInit
): Promise<Response> {
  const url = `${getApiBase()}${endpoint}`;
  return fetch(url, options);
}

/**
 * Format a number to fixed decimals, trimming trailing zeros
 */
export function formatNumber(value: number, decimals: number = 3): string {
  if (!Number.isFinite(value)) return String(value ?? '');
  const fixed = value.toFixed(decimals);
  return fixed.replace(/\.?0+$/, '');
}
