import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { CalibrationItem } from '../types';

interface Props {
  data: CalibrationItem[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload as CalibrationItem & { perfect: number };
  return (
    <div className="bg-white border border-gray-200 rounded p-2 text-sm shadow">
      <p className="font-medium mb-1">Confidence: {label}</p>
      <p className="text-blue-600">
        Actual accuracy: {(item.accuracy * 100).toFixed(1)}%
      </p>
      <p className="text-gray-400">
        Perfect calibration: {(item.perfect * 100).toFixed(1)}%
      </p>
      <p className="text-gray-500 text-xs mt-1">n = {item.n_games} games</p>
    </div>
  );
}

export default function CalibrationChart({ data }: Props) {
  const formatted = data.map((d) => ({
    ...d,
    // Midpoint of the bucket as the "expected" accuracy under perfect calibration
    perfect: (d.lo + Math.min(d.hi, 0.9)) / 2,
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={formatted} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
          <YAxis
            domain={[0.4, 1.0]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            verticalAlign="top"
            formatter={(value) =>
              value === 'accuracy' ? 'Actual accuracy' : 'Perfect calibration'
            }
            iconType="square"
            wrapperStyle={{ fontSize: 11 }}
          />
          <Bar dataKey="accuracy" fill="#3b82f6" radius={[3, 3, 0, 0]} name="accuracy" />
          <Line
            type="monotone"
            dataKey="perfect"
            stroke="#94a3b8"
            strokeDasharray="5 3"
            strokeWidth={2}
            dot={{ fill: '#94a3b8', r: 3, strokeWidth: 0 }}
            name="perfect"
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 text-center mt-1 italic">
        A well-calibrated model has bars matching the dashed line — predicted confidence equals observed accuracy.
      </p>
    </div>
  );
}
