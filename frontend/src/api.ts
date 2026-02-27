import type { CalibrationItem, FeatureImportanceItem, GameDetail, GameSummary, HomeAdvantageItem, SeasonInfo } from './types';

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

// Module-level cache for data that never changes during a session
let _calibrationCache: CalibrationItem[] | null = null;

export const api = {
  games: (date: string) => get<GameSummary[]>(`/predict/games?date=${date}`),
  game: (id: string) => get<GameDetail>(`/predict/game/${id}`),
  featureImportance: () => get<FeatureImportanceItem[]>('/analysis/feature-importance'),
  homeAdvantage: () => get<HomeAdvantageItem[]>('/analysis/home-advantage'),
  seasons: () => get<SeasonInfo[]>('/games/seasons'),
  calibration: async (): Promise<CalibrationItem[]> => {
    if (_calibrationCache) return _calibrationCache;
    _calibrationCache = await get<CalibrationItem[]>('/analysis/calibration');
    return _calibrationCache;
  },
};
