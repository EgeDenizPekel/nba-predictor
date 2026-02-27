import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import type { HomeAdvantageItem } from '../types';

interface Props {
  data: HomeAdvantageItem[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomDot(props: any) {
  const { cx, cy, payload } = props as {
    cx: number;
    cy: number;
    payload: HomeAdvantageItem;
  };
  if (!payload || cx == null || cy == null) return null;
  let fill = '#3b82f6';
  if (payload.is_bubble_season) fill = '#f97316';
  if (payload.is_no_fans_season) fill = '#ef4444';
  return <circle cx={cx} cy={cy} r={5} fill={fill} stroke="white" strokeWidth={2} />;
}

// Inline label for the bubble and no-fans seasons
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomLabel(props: any) {
  const { cx, cy, payload } = props as {
    cx: number;
    cy: number;
    payload: HomeAdvantageItem;
  };
  if (!payload || cx == null || cy == null) return null;
  if (!payload.is_bubble_season && !payload.is_no_fans_season) return null;
  const text = payload.is_bubble_season ? 'Bubble' : 'No fans';
  const color = payload.is_bubble_season ? '#f97316' : '#ef4444';
  return (
    <text x={cx} y={cy - 11} textAnchor="middle" fontSize={9} fill={color} fontWeight={600}>
      {text}
    </text>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as HomeAdvantageItem;
  const tag = d.is_bubble_season
    ? ' (bubble season)'
    : d.is_no_fans_season
    ? ' (no fans)'
    : '';
  return (
    <div className="bg-white border border-gray-200 rounded p-2 text-sm shadow">
      <p className="font-medium">{d.season}{tag}</p>
      <p>Home win rate: {(d.home_win_rate * 100).toFixed(1)}%</p>
      <p className="text-gray-500">{d.n_games} games</p>
    </div>
  );
}

export default function HomeAdvantageChart({ data }: Props) {
  // Pre-2010 average: seasons up to and including 2009-10 (exclude bubble)
  const pre2010 = data.filter(
    (d) => d.season <= '2009-10' && !d.is_bubble_season,
  );
  const pre2010Avg =
    pre2010.length > 0
      ? pre2010.reduce((s, d) => s + d.home_win_rate, 0) / pre2010.length
      : null;

  // Post-2018 average: 2018-19 onwards (exclude bubble)
  const post2018 = data.filter(
    (d) => d.season >= '2018-19' && !d.is_bubble_season,
  );
  const post2018Avg =
    post2018.length > 0
      ? post2018.reduce((s, d) => s + d.home_win_rate, 0) / post2018.length
      : null;

  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="season"
            tick={{ fontSize: 10 }}
            angle={-35}
            textAnchor="end"
            height={50}
          />
          <YAxis
            domain={[0.46, 0.68]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* 50% reference */}
          <ReferenceLine
            y={0.5}
            stroke="#94a3b8"
            strokeDasharray="4 4"
            label={{ value: '50%', position: 'insideTopRight', fontSize: 10, fill: '#94a3b8' }}
          />

          {/* Pre-2010 era average */}
          {pre2010Avg !== null && (
            <ReferenceLine
              y={pre2010Avg}
              stroke="#6366f1"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: `Pre-2010 avg ${(pre2010Avg * 100).toFixed(1)}%`,
                position: 'insideTopLeft',
                fontSize: 10,
                fill: '#6366f1',
              }}
            />
          )}

          {/* Post-2018 era average */}
          {post2018Avg !== null && (
            <ReferenceLine
              y={post2018Avg}
              stroke="#10b981"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: `Post-2018 avg ${(post2018Avg * 100).toFixed(1)}%`,
                position: 'insideBottomLeft',
                fontSize: 10,
                fill: '#10b981',
              }}
            />
          )}

          <Line
            type="monotone"
            dataKey="home_win_rate"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={<CustomDot />}
            label={<CustomLabel />}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-gray-500 justify-center">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-blue-500" /> Normal season
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-orange-500" /> Bubble (neutral site)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-red-500" /> No fans
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-5 border-t-2 border-dashed border-indigo-500" /> Pre-2010 avg
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-5 border-t-2 border-dashed border-emerald-500" /> Post-2018 avg
        </span>
      </div>
    </div>
  );
}
