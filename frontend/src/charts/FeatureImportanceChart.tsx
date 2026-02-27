import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { FeatureImportanceItem } from '../types';
import { fmtShapFeature } from '../utils';

interface Props {
  data: FeatureImportanceItem[];
  limit?: number;
}

// Feature group membership (bare key, no home_/away_ prefix)
const GROUP_KEYS: Record<string, string> = {
  win_pct_l5: 'Recent Form', win_pct_l10: 'Recent Form', win_pct_l20: 'Recent Form',
  off_rtg_l10: 'Efficiency', def_rtg_l10: 'Efficiency', pace_l10: 'Efficiency', ts_pct_l10: 'Efficiency',
  rest_days: 'Rest & Schedule', is_b2b: 'Rest & Schedule',
  season_win_pct: 'Season Standing', home_win_pct_season: 'Season Standing', away_win_pct_season: 'Season Standing',
  streak: 'Streak & Context', is_early_season: 'Streak & Context',
};

const GROUP_COLORS: Record<string, string> = {
  'Recent Form':     '#3b82f6', // blue
  'Efficiency':      '#8b5cf6', // purple
  'Rest & Schedule': '#f59e0b', // amber
  'Season Standing': '#10b981', // green
  'Streak & Context':'#f43f5e', // rose
};

function featureColor(featureName: string): string {
  const bare = featureName.replace(/^(home_|away_)/, '');
  const group = GROUP_KEYS[bare] ?? '';
  return GROUP_COLORS[group] ?? '#94a3b8';
}

export default function FeatureImportanceChart({ data, limit }: Props) {
  const formatted = [...data]
    .sort((a, b) => a.rank - b.rank)
    .slice(0, limit ?? data.length)
    .map((d) => ({ ...d, label: fmtShapFeature(d.feature) }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={formatted.length * 28 + 40}>
        <BarChart
          layout="vertical"
          data={formatted}
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(v: number) => v.toFixed(3)}
            tick={{ fontSize: 11 }}
            label={{ value: 'Mean |SHAP|', position: 'insideBottomRight', offset: -4, fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={130}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(v: number | undefined) => [(v ?? 0).toFixed(4), 'Mean |SHAP|']}
            labelFormatter={(label) => label}
          />
          <Bar dataKey="mean_abs_shap" radius={[0, 3, 3, 0]}>
            {formatted.map((d) => (
              <Cell key={d.feature} fill={featureColor(d.feature)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Group legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 justify-center">
        {Object.entries(GROUP_COLORS).map(([group, color]) => (
          <span key={group} className="flex items-center gap-1.5 text-xs text-gray-500">
            <span className="inline-block w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: color }} />
            {group}
          </span>
        ))}
      </div>
    </div>
  );
}
