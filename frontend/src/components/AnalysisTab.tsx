import { useEffect, useState } from 'react';
import type { FeatureImportanceItem, HomeAdvantageItem } from '../types';
import { api } from '../api';
import FeatureImportanceChart from '../charts/FeatureImportanceChart';
import HomeAdvantageChart from '../charts/HomeAdvantageChart';

export default function AnalysisTab() {
  const [importance, setImportance] = useState<FeatureImportanceItem[]>([]);
  const [homeAdv, setHomeAdv] = useState<HomeAdvantageItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.featureImportance(), api.homeAdvantage()])
      .then(([imp, adv]) => {
        setImportance(imp);
        setHomeAdv(adv);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-gray-400 text-sm text-center py-16">Loading…</p>;
  if (error) return <p className="text-red-500 text-sm text-center py-8">{error}</p>;

  return (
    <div className="flex flex-col gap-8">
      {/* Feature Importance */}
      <section>
        <h2 className="text-base font-semibold text-gray-800 mb-1">
          SHAP Feature Importance
        </h2>
        <p className="text-xs text-gray-500 mb-3">
          Mean absolute SHAP values from the XGBoost model on 2,000 test-season games.
          Higher = more influence on predictions. Prefix: H = home team, A = away team.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-y-auto">
          <FeatureImportanceChart data={importance} />
        </div>
      </section>

      {/* Home Advantage Trend */}
      <section>
        <h2 className="text-base font-semibold text-gray-800 mb-1">
          Home Advantage Trend (2003-04 to 2021-22)
        </h2>
        <p className="text-xs text-gray-500 mb-3">
          Season-by-season home win rate. Bubble games excluded from 2019-20 rate.
          Pre-2010 average ~60%; post-2020 average ~54%.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <HomeAdvantageChart data={homeAdv} />
        </div>
      </section>
    </div>
  );
}
