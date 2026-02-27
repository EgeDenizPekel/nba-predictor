export interface TeamFeatures {
  win_pct_l5: number | null;
  win_pct_l10: number | null;
  win_pct_l20: number | null;
  off_rtg_l10: number | null;
  def_rtg_l10: number | null;
  pace_l10: number | null;
  ts_pct_l10: number | null;
  rest_days: number | null;
  is_b2b: number;
  season_win_pct: number | null;
  home_win_pct_season: number | null;
  away_win_pct_season: number | null;
  streak: number;
  is_early_season: number;
}

export interface GameSummary {
  game_id: string;
  game_date: string;
  season: string;
  home_team_id: number;
  home_team_abbr: string;
  away_team_id: number;
  away_team_abbr: string;
  home_pts: number | null;
  away_pts: number | null;
  home_win: number;
  home_win_prob: number;
  is_bubble_game: number;
  is_no_fans_season: number;
}

export interface GameDetail extends GameSummary {
  home_features: TeamFeatures;
  away_features: TeamFeatures;
  shap_values: Record<string, number>; // feature_name -> SHAP value (positive = toward home win)
}

export interface FeatureImportanceItem {
  feature: string;
  mean_abs_shap: number;
  rank: number;
}

export interface HomeAdvantageItem {
  season: string;
  home_win_rate: number;
  n_games: number;
  is_bubble_season: boolean;
  is_no_fans_season: boolean;
}

export interface CalibrationItem {
  bucket: string;
  lo: number;
  hi: number;
  n_games: number;
  accuracy: number;
}

export interface SeasonInfo {
  season: string;
  min_date: string;
  max_date: string;
  n_games: number;
}
