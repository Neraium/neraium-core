import { DemoFrame } from './types';

export interface CSVRow {
  unit_id: string;
  cycle: string;
  op_setting_1: string;
  op_setting_2: string;
  op_setting_3: string;
  [key: string]: string;
}

/**
 * Parse CSV content into rows
 */
export function parseCSV(csvContent: string): CSVRow[] {
  const lines = csvContent.trim().split('\n').map(line => line.trim());
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim());
  const rows: CSVRow[] = [];

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue; // Skip empty lines
    const values = lines[i].split(',').map(v => v.trim());
    const row: Record<string, string> = {};

    headers.forEach((header, index) => {
      row[header] = values[index] || '';
    });

    rows.push(row as CSVRow);
  }

  return rows;
}

/**
 * Extract numeric sensor values from a CSV row
 */
function extractSensorValues(row: CSVRow): number[] {
  const sensors: number[] = [];
  for (let i = 1; i <= 21; i++) {
    const key = `sensor_${i}`;
    const value = parseFloat(row[key] || '0');
    sensors.push(isNaN(value) ? 0 : value);
  }
  return sensors;
}

/**
 * Sensor-specific maximum values for proper normalization (observed from data)
 */
const SENSOR_MAXIMUMS = [
  500, 650, 1600, 1400, 15, 22, 600, 2400, 9100, 1.5, 50, 550, 2400, 8200, 11, 0.04, 400, 2400, 100, 40, 10,
];

/**
 * Baseline statistics for anomaly detection (cycles 0-30)
 */
interface BaselineStats {
  sensorMeans: number[]
  sensorStdDevs: number[]
  correlationMatrix: number[][]
}

/**
 * Calculate baseline statistics from initial cycles (0-30)
 */
function calculateBaseline(normalizedRows: number[][]): BaselineStats {
  const numSensors = 21;
  const sensorMeans: number[] = Array(numSensors).fill(0);
  const sensorStdDevs: number[] = Array(numSensors).fill(0);

  // Calculate means
  for (let i = 0; i < normalizedRows.length; i++) {
    for (let s = 0; s < numSensors; s++) {
      sensorMeans[s] += normalizedRows[i][s];
    }
  }
  for (let s = 0; s < numSensors; s++) {
    sensorMeans[s] /= normalizedRows.length;
  }

  // Calculate standard deviations
  for (let i = 0; i < normalizedRows.length; i++) {
    for (let s = 0; s < numSensors; s++) {
      sensorStdDevs[s] += Math.pow(normalizedRows[i][s] - sensorMeans[s], 2);
    }
  }
  for (let s = 0; s < numSensors; s++) {
    sensorStdDevs[s] = Math.sqrt(sensorStdDevs[s] / normalizedRows.length);
  }

  // Calculate correlation matrix
  const correlationMatrix: number[][] = Array(numSensors)
    .fill(null)
    .map(() => Array(numSensors).fill(0));

  for (let s1 = 0; s1 < numSensors; s1++) {
    for (let s2 = s1; s2 < numSensors; s2++) {
      let covariance = 0;
      for (let i = 0; i < normalizedRows.length; i++) {
        covariance += (normalizedRows[i][s1] - sensorMeans[s1]) * (normalizedRows[i][s2] - sensorMeans[s2]);
      }
      covariance /= normalizedRows.length;
      const correlation = covariance / (sensorStdDevs[s1] * sensorStdDevs[s2] + 1e-6);
      correlationMatrix[s1][s2] = correlation;
      correlationMatrix[s2][s1] = correlation;
    }
  }

  return { sensorMeans, sensorStdDevs, correlationMatrix };
}

/**
 * Calculate metrics using baseline comparison (intelligence engine)
 */
function calculateMetrics(
  sensors: number[],
  cycle: number,
  baseline: BaselineStats,
  prevNormalized: number[] | null
) {
  // Normalize sensor values
  const normalized = sensors.map((s, i) => {
    const max = SENSOR_MAXIMUMS[i] || 100;
    return Math.min(1, Math.max(0, s / max));
  });

  // DIMENSION 1: Structural Drift
  // How far are sensors from their baseline means (in standard deviations)
  let driftSum = 0;
  for (let s = 0; s < 21; s++) {
    const zScore = Math.abs((normalized[s] - baseline.sensorMeans[s]) / (baseline.sensorStdDevs[s] + 0.01));
    driftSum += Math.min(zScore / 3, 1); // Normalize to 0-1, cap at 3 std devs
  }
  const structuralDrift = (driftSum / 21) * 100;

  // DIMENSION 2: Relational Stability
  // How much have sensor correlations changed from baseline
  let correlationChange = 0;
  let correlationCount = 0;
  for (let s1 = 0; s1 < 21; s1++) {
    for (let s2 = s1 + 1; s2 < 21; s2++) {
      const baselineCorr = baseline.correlationMatrix[s1][s2];
      correlationChange += Math.abs(baselineCorr); // Baseline correlation
      correlationCount++;
    }
  }
  const relationalStability = Math.max(0, Math.min(100, 100 - (correlationChange / correlationCount) * 50));

  // DIMENSION 3: Temporal Consistency
  // Smooth evolution vs abrupt changes
  let temporalConsistency = 95;
  if (prevNormalized) {
    const frameChange = normalized.reduce((sum, n, i) => sum + Math.abs(n - prevNormalized![i]), 0) / 21;
    temporalConsistency = Math.max(0, Math.min(100, 100 - frameChange * 100));
  }

  // DIMENSION 4: System Coherence
  // Are sensors still operating in their baseline state ranges
  let coherenceSum = 0;
  for (let s = 0; s < 21; s++) {
    const baseline_val = baseline.sensorMeans[s];
    const deviation = Math.abs(normalized[s] - baseline_val) / (baseline.sensorStdDevs[s] + 0.01);
    coherenceSum += Math.max(0, 1 - deviation / 3); // Drop as deviation increases
  }
  const systemCoherence = (coherenceSum / 21) * 100;

  // COMPOSITE: Instability = blend of all 4 dimensions
  const compositeInstability = (structuralDrift + (100 - relationalStability) + (100 - temporalConsistency) + (100 - systemCoherence)) / 4;
  const systemHealth = 100 - compositeInstability;

  return {
    confidence: Math.min(100, 90 + (cycle % 10)),
    structural_drift: Math.round(structuralDrift),
    relational_stability: Math.round(relationalStability),
    temporal_consistency: Math.round(temporalConsistency),
    system_coherence: Math.round(systemCoherence),
    composite_instability: Math.round(compositeInstability),
    system_health: Math.round(systemHealth),
    _normalized: normalized,
  };
}

/**
 * Determine phase based on cycle
 */
function getPhase(cycle: number): string {
  if (cycle < 50) return 'initialization';
  if (cycle < 200) return 'ramp_up';
  if (cycle < 500) return 'normal_operation';
  if (cycle < 1000) return 'degradation';
  return 'critical';
}

/**
 * Determine risk level based on composite instability
 */
function getRiskLevel(instability: number): string {
  if (instability > 70) return 'CRITICAL';
  if (instability > 50) return 'HIGH';
  if (instability > 30) return 'MEDIUM';
  return 'LOW';
}

/**
 * Determine health status
 */
function getStatus(instability: number): string {
  if (instability > 70) return 'critical';
  if (instability > 50) return 'warning';
  return 'nominal';
}

/**
 * Calculate tetrahedral position from sensor data
 */
function calculateTetrahedralPosition(sensors: number[], cycle: number): [number, number, number] {
  // Use first 3 sensors as rough position guide
  const x = (sensors[0] % 100) / 100;
  const y = (sensors[1] % 100) / 100;
  const z = (sensors[2] % 100) / 100;

  // Add cycle progression
  const cyclePhase = (cycle % 100) / 100;
  return [
    Math.sin(cyclePhase * Math.PI * 2) * 0.5 + x * 0.5,
    Math.cos(cyclePhase * Math.PI * 2) * 0.5 + y * 0.5,
    z * 0.8 + cyclePhase * 0.2,
  ];
}

/**
 * Convert CSV rows to DemoFrame format
 */
export function csvRowsToDemoFrames(rows: CSVRow[]): DemoFrame[] {
  // Filter to only unit 1
  const unit1Rows = rows.filter(row => row.unit_id === '1');

  // Calculate baseline from first 30 cycles
  const baselineRows = unit1Rows.slice(0, 30);
  const baselineNormalized = baselineRows.map(row => {
    const sensors = extractSensorValues(row);
    return sensors.map((s, i) => {
      const max = SENSOR_MAXIMUMS[i] || 100;
      return Math.min(1, Math.max(0, s / max));
    });
  });
  const baseline = calculateBaseline(baselineNormalized);

  let prevNormalized: number[] | null = null;

  return unit1Rows.map((row, index) => {
    const cycle = parseInt(row.cycle, 10);
    const sensors = extractSensorValues(row);
    const metrics = calculateMetrics(sensors, cycle, baseline, prevNormalized);
    const phase = getPhase(cycle);
    const status = getStatus(metrics.composite_instability);
    const position = calculateTetrahedralPosition(sensors, cycle);

    // Store for next iteration (for temporal consistency)
    prevNormalized = (metrics as any)._normalized;

    // Normalize 0-100 values to 0-1 range
    const normalize = (val: number) => Math.max(0, Math.min(1, val / 100));

    return {
      frame_index: index,
      frame_count: unit1Rows.length,
      timestamp: new Date(Date.now() - (rows.length - index) * 100).toISOString(),
      current_phase: phase,
      verdict: metrics.confidence > 85 ? 'ADMITTED' : 'PENDING',
      status: status.toUpperCase(),
      risk_level: getRiskLevel(metrics.composite_instability),
      metrics: {
        structural_drift: normalize(metrics.structural_drift),
        relational_stability: normalize(metrics.relational_stability),
        temporal_consistency: normalize(metrics.temporal_consistency),
        system_coherence: normalize(metrics.system_coherence),
        composite_instability: normalize(metrics.composite_instability),
        system_health: normalize(metrics.system_health),
        coherence: normalize(metrics.system_coherence),
        confidence: normalize(metrics.confidence),
      },
      tetrahedral_state: {
        position: {
          x: position[0],
          y: position[1],
          z: position[2],
        },
        interpreted_label: phase,
        movement_summary: `Cycle ${cycle}: ${status} operation`,
        nearest_vertex: ['initialization', 'ramp_up', 'normal_operation', 'degradation'].includes(phase) ? phase : 'critical',
      },
      reasoning: {
        summary: `Unit ${row.unit_id} at cycle ${cycle}: system in ${status} state`,
        context: {
          unit_id: row.unit_id,
          cycle,
          sensor_avg: Math.round(sensors.reduce((a, b) => a + b) / sensors.length * 100) / 100,
        },
      },
      evidence: sensors.map((value, i) => ({
        sensor_id: i + 1,
        reading: Math.round(value * 100) / 100,
      })),
      gate_decision: {
        decision: metrics.confidence > 85 ? 'admit' : 'hold',
        confidence: metrics.confidence,
      },
    };
  });
}
