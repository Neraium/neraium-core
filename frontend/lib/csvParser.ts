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
 * Calculate the 4 tetrahedron dimensions from sensor readings
 */
function calculateMetrics(
  sensors: number[],
  cycle: number,
  prevMetrics?: { normalized: number[]; structuralDrift: number } | null
) {
  // Normalize sensor values to 0-1 range using sensor-specific maximums
  const normalized = sensors.map((s, i) => {
    const max = SENSOR_MAXIMUMS[i] || 100;
    return Math.min(1, Math.max(0, s / max));
  });

  // TETRAHEDRON DIMENSION 1: Structural Drift
  // Distance from sensor baseline (optimal ~0.45-0.55 range per sensor)
  const baselineDeviations = normalized.map(n => Math.abs(n - 0.5));
  const structuralDrift = (baselineDeviations.reduce((a, b) => a + b, 0) / baselineDeviations.length) * 100;

  // TETRAHEDRON DIMENSION 2: Relational Stability
  // How consistent are sensor relationships (correlation stability)
  // Calculate if sensors are moving together or diverging
  const sensorVariance = normalized.reduce((sum, n) => sum + Math.pow(n - 0.5, 2), 0) / normalized.length;
  const relationalStability = Math.max(0, Math.min(100, 100 - Math.sqrt(sensorVariance) * 100));

  // TETRAHEDRON DIMENSION 3: Temporal Consistency
  // How smoothly is the system evolving (vs breaking patterns)
  let temporalConsistency = 85; // Default for first frame
  if (prevMetrics) {
    const driftChange = Math.abs(structuralDrift - prevMetrics.structuralDrift);
    // Smooth change = high consistency, rapid change = low consistency
    temporalConsistency = Math.max(0, Math.min(100, 100 - (driftChange * 1.5)));
  }

  // TETRAHEDRON DIMENSION 4: System Coherence
  // Overall "tightness" - how concentrated are sensor values
  // High entropy (spread out) = low coherence, Low entropy (clustered) = high coherence
  const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
  const entropy = -normalized
    .map(n => {
      const p = n > 0 && n < 1 ? n : 0.5;
      return p > 0 ? -p * Math.log2(p) : 0;
    })
    .reduce((a, b) => a + b, 0);
  const systemCoherence = Math.max(0, Math.min(100, 100 - (entropy * 10)));

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
    // Store for next iteration
    _normalized: normalized,
    _structuralDrift: structuralDrift,
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

  let prevMetrics: { normalized: number[]; structuralDrift: number } | null = null;

  return unit1Rows.map((row, index) => {
    const cycle = parseInt(row.cycle, 10);
    const sensors = extractSensorValues(row);
    const metrics = calculateMetrics(sensors, cycle, prevMetrics);
    const phase = getPhase(cycle);
    const status = getStatus(metrics.composite_instability);
    const position = calculateTetrahedralPosition(sensors, cycle);

    // Store for next iteration (for temporal consistency)
    prevMetrics = {
      normalized: (metrics as any)._normalized,
      structuralDrift: (metrics as any)._structuralDrift,
    };

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
