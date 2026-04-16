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
  500,    // S1: ~518
  650,    // S2: ~642
  1600,   // S3: ~1585
  1400,   // S4: ~1404
  15,     // S5: ~14.6
  22,     // S6: ~21.6
  600,    // S7: ~553
  2400,   // S8: ~2387
  9100,   // S9: ~9053
  1.5,    // S10: ~1.3
  50,     // S11: ~47
  550,    // S12: ~521
  2400,   // S13: ~2388
  8200,   // S14: ~8136
  11,     // S15: ~10.9
  0.04,   // S16: ~0.03
  400,    // S17: ~392
  2400,   // S18: ~2388
  100,    // S19: ~100
  40,     // S20: ~39
  10,     // S21: ~9
];

/**
 * Calculate health metrics from sensor readings
 */
function calculateMetrics(sensors: number[], cycle: number, maxCycle: number) {
  // Normalize sensor values to 0-1 range using sensor-specific maximums
  const normalized = sensors.map((s, i) => {
    const max = SENSOR_MAXIMUMS[i] || 100;
    return Math.min(1, Math.max(0, s / max));
  });

  // Calculate different aspects of health
  const avgSensors = normalized.reduce((a, b) => a + b, 0) / normalized.length;
  const variance = normalized.reduce((sum, val) => sum + Math.pow(val - avgSensors, 2), 0) / normalized.length;
  const stability = 100 - (Math.sqrt(variance) * 100); // Lower std dev = higher stability

  const cycleRatio = Math.max(0, Math.min(1, cycle / Math.max(maxCycle, 1)));
  const structuralDrift = 15 + cycleRatio * 65 + variance * 15;
  const relationalStability = 96 - cycleRatio * 50 - variance * 20;
  const coherence = (relationalStability * 0.55) + ((100 - structuralDrift) * 0.45);
  const confidence = 72 + (coherence * 0.28) + ((100 - structuralDrift) * 0.08);

  return {
    confidence: Math.max(0, Math.min(100, confidence)),
    structural_drift: Math.max(0, Math.min(100, structuralDrift)),
    relational_stability: Math.max(0, Math.min(100, relationalStability)),
    coherence: Math.max(0, Math.min(100, coherence)),
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
 * Determine health status
 */
function getStatus(drift: number, stability: number): string {
  if (drift >= 78 || stability <= 45) return 'critical';
  if (drift >= 55 || stability <= 65) return 'warning';
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
  const maxCycle = unit1Rows.reduce((max, row) => Math.max(max, parseInt(row.cycle, 10) || 0), 0);

  return unit1Rows.map((row, index) => {
    const cycle = parseInt(row.cycle, 10);
    const sensors = extractSensorValues(row);
    const metrics = calculateMetrics(sensors, cycle, maxCycle);
    const phase = getPhase(cycle);
    const status = getStatus(metrics.structural_drift, metrics.relational_stability);
    const position = calculateTetrahedralPosition(sensors, cycle);

    return {
      frame_index: index,
      frame_count: unit1Rows.length,
      timestamp: new Date(Date.now() - (rows.length - index) * 100).toISOString(),
      current_phase: phase,
      verdict: metrics.confidence > 85 ? 'ADMITTED' : 'PENDING',
      status: status.toUpperCase(),
      metrics: {
        structural_drift: Math.round((metrics.structural_drift / 100) * 100) / 100,
        relational_stability: Math.round((metrics.relational_stability / 100) * 100) / 100,
        coherence: Math.round((metrics.coherence / 100) * 100) / 100,
        confidence: Math.round((metrics.confidence / 100) * 100) / 100,
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
