'use client';

import React from 'react';

interface EtaRange {
  min: number;
  max: number;
  unit: string;
}

interface FuturePath {
  path_id: string;
  label: string;
  probability: number;
  risk: string;
  eta_range: EtaRange;
  trend_summary: string;
  drivers: string[];
  point_of_no_return: number | null;
}

interface Intervention {
  action: string;
  expected_effect: string;
  target_paths: string[];
  confidence: number;
}

interface CurrentState {
  regime_name: string | null;
  health_band: string;
  confidence: number;
  summary: string;
  structural_drift: number;
  relational_stability: number;
  temporal_consistency: number;
}

interface FuturePathMapData {
  current_state: CurrentState;
  future_paths: FuturePath[];
  recommended_interventions: Intervention[];
}

interface FuturePathMapProps {
  data: FuturePathMapData | null;
  isLoading?: boolean;
  error?: string | null;
}

const getRiskColor = (risk: string): string => {
  switch (risk) {
    case 'low':
      return '#10b981';
    case 'medium':
      return '#f59e0b';
    case 'high':
      return '#ef4444';
    default:
      return '#6b7280';
  }
};

const getHealthBandColor = (band: string): string => {
  switch (band) {
    case 'stable':
      return '#10b981';
    case 'watch':
      return '#f59e0b';
    case 'critical':
      return '#ef4444';
    default:
      return '#6b7280';
  }
};

const CurrentStateCard: React.FC<{ state: CurrentState }> = ({ state }) => {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm mb-6">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Current System State</h3>
          {state.regime_name && (
            <p className="text-sm text-gray-500 mt-1">Regime: {state.regime_name}</p>
          )}
        </div>
        <div
          className="px-3 py-1 rounded-full text-sm font-medium text-white"
          style={{ backgroundColor: getHealthBandColor(state.health_band) }}
        >
          {state.health_band.toUpperCase()}
        </div>
      </div>

      <p className="text-gray-700 text-sm mb-4">{state.summary}</p>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <div className="text-xs font-medium text-gray-500 uppercase mb-1">
            Structural Drift
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-gray-900">
              {(state.structural_drift * 100).toFixed(1)}%
            </span>
            <div className="w-12 h-1 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full"
                style={{
                  width: `${state.structural_drift * 100}%`,
                  backgroundColor: getHealthBandColor(state.health_band),
                }}
              />
            </div>
          </div>
        </div>

        <div>
          <div className="text-xs font-medium text-gray-500 uppercase mb-1">
            Relational Stability
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-gray-900">
              {(state.relational_stability * 100).toFixed(1)}%
            </span>
            <div className="w-12 h-1 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500"
                style={{ width: `${state.relational_stability * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div>
          <div className="text-xs font-medium text-gray-500 uppercase mb-1">
            Confidence
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-gray-900">
              {(state.confidence * 100).toFixed(0)}%
            </span>
            <div className="text-xs text-gray-500">in state</div>
          </div>
        </div>
      </div>
    </div>
  );
};

const PathCard: React.FC<{ path: FuturePath }> = ({ path }) => {
  const probability = path.probability * 100;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <h4 className="font-semibold text-gray-900">{path.label}</h4>
        <span
          className="px-2.5 py-1 rounded text-xs font-medium text-white"
          style={{ backgroundColor: getRiskColor(path.risk) }}
        >
          {path.risk.toUpperCase()} RISK
        </span>
      </div>

      <div className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium text-gray-700">Probability</span>
          <span className="text-sm font-bold text-gray-900">{probability.toFixed(1)}%</span>
        </div>
        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-300"
            style={{
              width: `${probability}%`,
              backgroundColor: getRiskColor(path.risk),
            }}
          />
        </div>
      </div>

      <div className="bg-gray-50 rounded p-3 mb-4">
        <p className="text-sm text-gray-700">{path.trend_summary}</p>
      </div>

      <div className="mb-4">
        <div className="text-xs font-medium text-gray-500 uppercase mb-2">
          ETA to Critical State
        </div>
        <div className="text-sm font-semibold text-gray-900">
          {path.eta_range.min}–{path.eta_range.max} {path.eta_range.unit}
        </div>
      </div>

      {path.point_of_no_return !== null && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded">
          <div className="text-xs font-medium text-red-800 uppercase mb-1">
            Point of No Return
          </div>
          <div className="text-sm font-semibold text-red-900">
            ~{path.point_of_no_return} cycles
          </div>
        </div>
      )}

      <div>
        <div className="text-xs font-medium text-gray-500 uppercase mb-2">
          Key Drivers
        </div>
        <div className="flex flex-wrap gap-2">
          {path.drivers.map((driver, idx) => (
            <span
              key={idx}
              className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
            >
              {driver}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

const InterventionCard: React.FC<{ intervention: Intervention }> = ({
  intervention,
}) => {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-semibold text-blue-900">{intervention.action}</h4>
        <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded">
          {(intervention.confidence * 100).toFixed(0)}% confident
        </span>
      </div>

      <p className="text-sm text-blue-800 mb-3">{intervention.expected_effect}</p>

      <div>
        <div className="text-xs font-medium text-blue-700 uppercase mb-2">
          Targets
        </div>
        <div className="flex flex-wrap gap-2">
          {intervention.target_paths.map((pathId) => (
            <span
              key={pathId}
              className="px-2 py-1 bg-blue-200 text-blue-900 text-xs rounded"
            >
              {pathId}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export const FuturePathMap: React.FC<FuturePathMapProps> = ({
  data,
  isLoading = false,
  error = null,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
        <div className="inline-flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-400 rounded-full animate-spin" />
          <span className="text-gray-600">Analyzing system trajectories...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
        <div className="font-medium mb-1">Unable to generate Future Path Map</div>
        <div className="text-sm">{error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
        <div className="text-gray-500">No path analysis available</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Future Path Map</h2>
        <p className="text-gray-600">
          Inferred trajectories from current system state and recommended interventions
        </p>
      </div>

      {/* Current State */}
      <CurrentStateCard state={data.current_state} />

      {/* Paths Section */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Plausible System Trajectories
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.future_paths.map((path) => (
            <PathCard key={path.path_id} path={path} />
          ))}
        </div>
      </div>

      {/* Interventions Section */}
      {data.recommended_interventions.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Recommended Interventions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.recommended_interventions.map((intervention, idx) => (
              <InterventionCard key={idx} intervention={intervention} />
            ))}
          </div>
        </div>
      )}

      {/* Notes */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
        <div className="font-medium text-gray-900 mb-1">How to interpret this map:</div>
        <ul className="list-disc list-inside space-y-1">
          <li>
            The <strong>probability</strong> indicates how likely each path is from the current state
          </li>
          <li>
            The <strong>ETA</strong> estimates when critical conditions might be reached
          </li>
          <li>
            The <strong>point of no return</strong> marks when intervention becomes difficult
          </li>
          <li>
            <strong>Interventions</strong> can shift the system toward safer paths
          </li>
        </ul>
      </div>
    </div>
  );
};

export default FuturePathMap;
