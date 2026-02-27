import { Fragment, useEffect, useState } from 'react';
import type { CalibrationItem, GameDetail as GameDetailType, TeamFeatures } from '../types';
import { api } from '../api';
import { FEATURE_LABELS, fmtFeature, fmtShapFeature } from '../utils';

interface Props {
  gameId: string;
  homeAbbr: string;
  awayAbbr: string;
}

// ---------------------------------------------------------------------------
// Feature table config
// ---------------------------------------------------------------------------

const FEATURE_DESCRIPTIONS: Record<keyof TeamFeatures, string> = {
  win_pct_l5: 'Win rate over the last 5 games',
  win_pct_l10: 'Win rate over the last 10 games',
  win_pct_l20: 'Win rate over the last 20 games',
  off_rtg_l10: 'Points scored per 100 possessions — last 10 games',
  def_rtg_l10: 'Points allowed per 100 possessions — lower is better — last 10 games',
  pace_l10: 'Possessions per 48 minutes — last 10 games',
  ts_pct_l10: 'Shooting efficiency accounting for 2s, 3s, and free throws — last 10 games',
  rest_days: 'Days since the team last played',
  is_b2b: 'Playing the second game on back-to-back nights',
  season_win_pct: 'Overall win percentage for the current season',
  home_win_pct_season: "This team's win rate at their home venue this season",
  away_win_pct_season: "This team's win rate on the road this season",
  streak: 'Current streak — positive means win streak, negative means losing streak',
  is_early_season: 'First 15 games of the season — rolling windows are still filling up',
};

const FEATURE_GROUPS: { label: string; keys: (keyof TeamFeatures)[] }[] = [
  { label: 'Recent Form', keys: ['win_pct_l5', 'win_pct_l10', 'win_pct_l20'] },
  { label: 'Efficiency', keys: ['off_rtg_l10', 'def_rtg_l10', 'pace_l10', 'ts_pct_l10'] },
  { label: 'Rest & Schedule', keys: ['rest_days', 'is_b2b'] },
  { label: 'Season Standing', keys: ['season_win_pct', 'home_win_pct_season', 'away_win_pct_season'] },
  { label: 'Streak & Context', keys: ['streak', 'is_early_season'] },
];

// Typical range per feature (used to normalize gap for dimming near-zero rows)
const FEATURE_RANGE: Partial<Record<keyof TeamFeatures, number>> = {
  win_pct_l5: 1, win_pct_l10: 1, win_pct_l20: 1,
  off_rtg_l10: 20, def_rtg_l10: 20, pace_l10: 10, ts_pct_l10: 0.15,
  rest_days: 5, season_win_pct: 1,
  home_win_pct_season: 1, away_win_pct_season: 1,
  streak: 10,
};

// Returns true if the feature difference is small enough to dim
function isNearEqual(key: keyof TeamFeatures, home: number | null, away: number | null): boolean {
  if (home === null || away === null) return false;
  const range = FEATURE_RANGE[key];
  if (!range) return false; // binary features never dim
  return Math.abs(home - away) / range < 0.05;
}

const HIGHER_BETTER = new Set<keyof TeamFeatures>([
  'win_pct_l5', 'win_pct_l10', 'win_pct_l20',
  'off_rtg_l10', 'ts_pct_l10',
  'rest_days', 'season_win_pct', 'home_win_pct_season', 'away_win_pct_season',
  'streak',
]);
const LOWER_BETTER = new Set<keyof TeamFeatures>(['def_rtg_l10', 'is_b2b']);

function advantage(
  key: keyof TeamFeatures,
  home: number | null,
  away: number | null,
): 'home' | 'away' | 'none' {
  if (home === null || away === null || home === away) return 'none';
  if (HIGHER_BETTER.has(key)) return home > away ? 'home' : 'away';
  if (LOWER_BETTER.has(key)) return home < away ? 'home' : 'away';
  return 'none';
}

// ---------------------------------------------------------------------------
// Local SHAP chart (diverging bars, pure CSS)
// ---------------------------------------------------------------------------

function LocalSHAPChart({ shapValues }: { shapValues: Record<string, number> }) {
  const entries = Object.entries(shapValues)
    .map(([feature, value]) => ({ feature, label: fmtShapFeature(feature), value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 12);

  const maxAbs = Math.max(...entries.map((e) => Math.abs(e.value)), 0.001);

  return (
    <div>
      <div className="flex items-center justify-between text-xs text-gray-400 mb-2 px-1">
        <span className="text-blue-500 font-medium">← Pushes home win</span>
        <span className="text-amber-500 font-medium">Pushes away win →</span>
      </div>
      <div className="flex flex-col gap-1" role="img" aria-label="SHAP contribution chart">
        {entries.map(({ feature, label, value }) => {
          const pct = (Math.abs(value) / maxAbs) * 44; // max fill = 44% of half-width
          const isHome = value > 0;
          return (
            <div key={feature} className="flex items-center gap-2 text-xs">
              <div className="w-32 text-right text-gray-500 shrink-0 truncate" title={label}>
                {label}
              </div>
              <div className="flex-1 relative h-4">
                {/* Center line */}
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-200 z-10" />
                {/* Bar */}
                {isHome ? (
                  <div
                    className="absolute top-1 bottom-1 right-1/2 bg-blue-400 rounded-l"
                    style={{ width: `${pct}%` }}
                  />
                ) : (
                  <div
                    className="absolute top-1 bottom-1 left-1/2 bg-amber-400 rounded-r"
                    style={{ width: `${pct}%` }}
                  />
                )}
              </div>
              <div
                className={`w-14 text-right tabular-nums shrink-0 font-mono ${
                  isHome ? 'text-blue-600' : 'text-amber-600'
                }`}
              >
                {value > 0 ? '+' : ''}
                {value.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-gray-400 mt-3 text-center italic">
        Values are additive contributions to the log-odds of home win
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Calibration context
// ---------------------------------------------------------------------------

function CalibrationContext({
  calibration,
  homeWinProb,
}: {
  calibration: CalibrationItem[];
  homeWinProb: number;
}) {
  // The current game's confidence bucket
  const confidence = Math.max(homeWinProb, 1 - homeWinProb);
  const activeBucket = calibration.find(
    (c) => confidence >= c.lo && confidence < c.hi,
  );

  return (
    <div className="space-y-1.5">
      <p className="text-xs text-gray-400 mb-2">
        When the model is this confident, how often has it been right?
        Based on 2020-21 + 2021-22 test seasons.
      </p>
      {calibration.map((item) => {
        const isActive = item === activeBucket;
        const barWidth = item.accuracy * 100;
        return (
          <div
            key={item.bucket}
            className={`flex items-center gap-2 text-xs rounded px-2 py-1 ${
              isActive ? 'bg-blue-50 ring-1 ring-blue-200' : ''
            }`}
          >
            <div className="w-14 shrink-0 font-medium text-gray-600">{item.bucket}</div>
            <div className="flex-1 relative h-2 bg-gray-100 rounded-full overflow-visible">
              <div
                className={`h-full rounded-full ${isActive ? 'bg-blue-500' : 'bg-gray-300'}`}
                style={{ width: `${barWidth}%` }}
              />
              {isActive && (
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-600 border-2 border-white shadow-sm"
                  style={{ left: `calc(${barWidth}% - 6px)` }}
                />
              )}
            </div>
            <div className="w-10 text-right tabular-nums text-gray-700 font-medium">
              {(item.accuracy * 100).toFixed(0)}%
            </div>
            <div className="w-16 text-right tabular-nums text-gray-400">
              n={item.n_games}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function GameDetail({ gameId, homeAbbr, awayAbbr }: Props) {
  const [detail, setDetail] = useState<GameDetailType | null>(null);
  const [calibration, setCalibration] = useState<CalibrationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setDetail(null);
    Promise.all([api.game(gameId), api.calibration()])
      .then(([gameData, calData]) => {
        setDetail(gameData);
        setCalibration(calData);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [gameId]);

  if (loading)
    return <p className="text-gray-500 text-sm py-8 text-center">Loading features…</p>;
  if (error)
    return <p className="text-red-500 text-sm py-4 text-center">{error}</p>;
  if (!detail) return null;

  return (
    <div className="flex gap-5 items-start">
      {/* Left: feature breakdown table */}
      <div className="w-96 shrink-0 border border-gray-200 rounded-lg">
        <div>
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="px-4 py-2.5 text-left text-xs font-semibold text-blue-600 w-1/3">
                  {homeAbbr} (home)
                </th>
                <th className="px-4 py-2.5 text-center text-xs font-medium text-gray-400 uppercase tracking-wide">
                  Feature
                </th>
                <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 w-1/3">
                  {awayAbbr} (away)
                </th>
              </tr>
            </thead>
            <tbody>
              {FEATURE_GROUPS.map(({ label, keys }) => (
                <Fragment key={label}>
                  <tr className="bg-gray-100 border-t border-b border-gray-200">
                    <td
                      colSpan={3}
                      className="px-4 py-1 text-xs font-semibold text-gray-500 uppercase tracking-wider"
                    >
                      {label}
                    </td>
                  </tr>
                  {keys.map((key) => {
                    const homeVal = detail.home_features[key] as number | null;
                    const awayVal = detail.away_features[key] as number | null;
                    const adv = advantage(key, homeVal, awayVal);
                    const dim = isNearEqual(key, homeVal, awayVal);
                    return (
                      <tr key={key} className={`border-t border-gray-100 group/row ${dim ? 'opacity-40' : ''}`}>
                        <td
                          className={`px-4 py-1.5 font-medium text-blue-700 ${adv === 'home' ? 'bg-green-50' : ''}`}
                        >
                          {fmtFeature(key, homeVal)}
                        </td>
                        <td className="px-4 py-1.5 text-center relative">
                          <span className="text-gray-400 text-xs cursor-default group-hover/row:text-gray-600 transition-colors">
                            {FEATURE_LABELS[key]}
                          </span>
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover/row:block z-50 pointer-events-none">
                            <div className="bg-gray-800 text-white text-xs rounded px-2.5 py-1.5 whitespace-nowrap shadow-lg">
                              {FEATURE_DESCRIPTIONS[key]}
                            </div>
                            <div className="w-2 h-2 bg-gray-800 rotate-45 mx-auto -mt-1" />
                          </div>
                        </td>
                        <td
                          className={`px-4 py-1.5 text-right font-medium text-gray-700 ${adv === 'away' ? 'bg-green-50' : ''}`}
                        >
                          {fmtFeature(key, awayVal)}
                        </td>
                      </tr>
                    );
                  })}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right: SHAP + calibration */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        {/* Why this prediction - local SHAP */}
        <div className="border border-gray-200 rounded-lg p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Why This Prediction
          </h3>
          <LocalSHAPChart shapValues={detail.shap_values} />
        </div>

        {/* Model confidence context - calibration */}
        {calibration.length > 0 && (
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Model Confidence Context
            </h3>
            <CalibrationContext
              calibration={calibration}
              homeWinProb={detail.home_win_prob}
            />
          </div>
        )}
      </div>
    </div>
  );
}
