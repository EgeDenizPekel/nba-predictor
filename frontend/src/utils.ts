import type { TeamFeatures } from './types';

/** Full team names keyed by abbreviation, including historical variants (2003-04 through 2021-22). */
export const TEAM_NAMES: Record<string, string> = {
  ATL: 'Atlanta Hawks',
  BOS: 'Boston Celtics',
  BKN: 'Brooklyn Nets',
  CHA: 'Charlotte Bobcats',
  CHO: 'Charlotte Hornets',
  CHI: 'Chicago Bulls',
  CLE: 'Cleveland Cavaliers',
  DAL: 'Dallas Mavericks',
  DEN: 'Denver Nuggets',
  DET: 'Detroit Pistons',
  GSW: 'Golden State Warriors',
  HOU: 'Houston Rockets',
  IND: 'Indiana Pacers',
  LAC: 'LA Clippers',
  LAL: 'Los Angeles Lakers',
  MEM: 'Memphis Grizzlies',
  MIA: 'Miami Heat',
  MIL: 'Milwaukee Bucks',
  MIN: 'Minnesota Timberwolves',
  NJN: 'New Jersey Nets',
  NOH: 'New Orleans Hornets',
  NOK: 'New Orleans/OKC Hornets',
  NOP: 'New Orleans Pelicans',
  NYK: 'New York Knicks',
  OKC: 'Oklahoma City Thunder',
  ORL: 'Orlando Magic',
  PHI: 'Philadelphia 76ers',
  PHX: 'Phoenix Suns',
  POR: 'Portland Trail Blazers',
  SAC: 'Sacramento Kings',
  SAS: 'San Antonio Spurs',
  SEA: 'Seattle SuperSonics',
  TOR: 'Toronto Raptors',
  UTA: 'Utah Jazz',
  WAS: 'Washington Wizards',
};

export function fmtTeamName(abbr: string): string {
  return TEAM_NAMES[abbr] ?? abbr;
}

export function fmtPct(v: number | null): string {
  if (v === null) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

export function fmtRtg(v: number | null): string {
  if (v === null) return '—';
  return v.toFixed(1);
}

export function fmtDays(v: number | null): string {
  if (v === null) return '—';
  const d = Math.round(v);
  return d === 1 ? '1 day' : `${d} days`;
}

export function fmtStreak(v: number): string {
  if (v > 0) return `+${v} W`;
  if (v < 0) return `${v} L`;
  return '0';
}

export function fmtBool(v: number): string {
  return v === 1 ? 'Yes' : 'No';
}

/** Human-readable label for a raw feature name (with prefix stripped). */
export const FEATURE_LABELS: Record<keyof TeamFeatures, string> = {
  win_pct_l5: 'Win% (L5)',
  win_pct_l10: 'Win% (L10)',
  win_pct_l20: 'Win% (L20)',
  off_rtg_l10: 'Off Rtg (L10)',
  def_rtg_l10: 'Def Rtg (L10)',
  pace_l10: 'Pace (L10)',
  ts_pct_l10: 'TS% (L10)',
  rest_days: 'Rest Days',
  is_b2b: 'Back-to-Back',
  season_win_pct: 'Season Win%',
  home_win_pct_season: 'Home Venue Win%',
  away_win_pct_season: 'Away Venue Win%',
  streak: 'Streak',
  is_early_season: 'Early Season',
};

/** Format a single TeamFeatures value for display in the detail table. */
export function fmtFeature(key: keyof TeamFeatures, val: number | null): string {
  if (val === null) return '—';
  switch (key) {
    case 'win_pct_l5':
    case 'win_pct_l10':
    case 'win_pct_l20':
    case 'season_win_pct':
    case 'home_win_pct_season':
    case 'away_win_pct_season':
    case 'ts_pct_l10':
      return fmtPct(val);
    case 'off_rtg_l10':
    case 'def_rtg_l10':
    case 'pace_l10':
      return val.toFixed(1);
    case 'rest_days':
      return fmtDays(val);
    case 'is_b2b':
    case 'is_early_season':
      return fmtBool(val);
    case 'streak':
      return fmtStreak(val);
    default:
      return String(val);
  }
}

/** Strip home_/away_ prefix from a SHAP feature name and make it readable. */
export function fmtShapFeature(name: string): string {
  const stripped = name.replace(/^(home_|away_)/, '');
  const prefix = name.startsWith('home_') ? 'H ' : name.startsWith('away_') ? 'A ' : '';
  const label = (FEATURE_LABELS as Record<string, string>)[stripped] ?? stripped;
  return `${prefix}${label}`;
}
