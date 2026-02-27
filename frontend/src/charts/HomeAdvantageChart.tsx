import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
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
  let fill = '#3b82f6';
  if (payload.is_bubble_season) fill = '#f97316';
  if (payload.is_no_fans_season) fill = '#ef4444';
  return <circle cx={cx} cy={cy} r={5} fill={fill} stroke="white" strokeWidth={2} />;
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
  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
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
          <ReferenceLine
            y={0.5}
            stroke="#94a3b8"
            strokeDasharray="4 4"
            label={{ value: '50%', position: 'insideTopRight', fontSize: 10, fill: '#94a3b8' }}
          />
          <Line
            type="monotone"
            dataKey="home_win_rate"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={<CustomDot />}
          />
          <Legend
            verticalAlign="top"
            formatter={() => 'Home win rate'}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-2 text-xs text-gray-500 justify-center">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-blue-500" /> Normal season
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-orange-500" /> Bubble season
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-red-500" /> No fans
        </span>
      </div>
    </div>
  );
}
