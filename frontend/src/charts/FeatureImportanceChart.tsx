import {
  BarChart,
  Bar,
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
}

export default function FeatureImportanceChart({ data }: Props) {
  const formatted = [...data]
    .sort((a, b) => a.rank - b.rank)
    .map((d) => ({ ...d, label: fmtShapFeature(d.feature) }));

  return (
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
        <Bar dataKey="mean_abs_shap" fill="#3b82f6" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
