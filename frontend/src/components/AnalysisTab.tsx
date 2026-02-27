import { useEffect, useState } from 'react';
import type { CalibrationItem, FeatureImportanceItem, HomeAdvantageItem } from '../types';
import { api } from '../api';
import FeatureImportanceChart from '../charts/FeatureImportanceChart';
import HomeAdvantageChart from '../charts/HomeAdvantageChart';
import CalibrationChart from '../charts/CalibrationChart';

export default function AnalysisTab() {
  const [importance, setImportance] = useState<FeatureImportanceItem[]>([]);
  const [homeAdv, setHomeAdv] = useState<HomeAdvantageItem[]>([]);
  const [calibration, setCalibration] = useState<CalibrationItem[]>([]);
  const [showAllFeatures, setShowAllFeatures] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.featureImportance(), api.homeAdvantage(), api.calibration()])
      .then(([imp, adv, cal]) => {
        setImportance(imp);
        setHomeAdv(adv);
        setCalibration(cal);
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
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="text-base font-semibold text-gray-800">
            SHAP Feature Importance
          </h2>
          <button
            onClick={() => setShowAllFeatures((v) => !v)}
            className="text-xs text-blue-600 hover:text-blue-800 font-medium transition-colors"
          >
            {showAllFeatures ? 'Show top 10' : `Show all ${importance.length}`}
          </button>
        </div>
        <p className="text-xs text-gray-500 mb-3">
          Mean absolute SHAP values from the XGBoost model on 2,000 test-season games.
          Higher = more influence on predictions. Prefix: H = home team, A = away team.
          Colors indicate feature group.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-y-auto">
          <FeatureImportanceChart data={importance} limit={showAllFeatures ? undefined : 10} />
        </div>
      </section>

      {/* Home Advantage Trend */}
      <section>
        <h2 className="text-base font-semibold text-gray-800 mb-1">
          Home Advantage Trend (2003-04 to 2021-22)
        </h2>
        <p className="text-xs text-gray-500 mb-3">
          Season-by-season home win rate. Bubble games (neutral site) excluded from 2019-20.
          Reference lines show the pre-2010 and post-2018 era averages — the ~6pp decline is structural, not noise.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <HomeAdvantageChart data={homeAdv} />
        </div>
      </section>

      {/* Calibration reliability diagram */}
      {calibration.length > 0 && (
        <section>
          <h2 className="text-base font-semibold text-gray-800 mb-1">
            Calibration Reliability
          </h2>
          <p className="text-xs text-gray-500 mb-3">
            When the model says 70%, does it win 70% of the time? Bars show observed accuracy
            per confidence bucket on the 2020-21 + 2021-22 test seasons (~2,300 games).
            The dashed line is perfect calibration.
          </p>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <CalibrationChart data={calibration} />
          </div>
        </section>
      )}
    </div>
  );
}
