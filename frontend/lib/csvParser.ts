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
 * Calculate health metrics from sensor readings
 */
function calculateMetrics(sensors: number[], cycle: number) {
  // Normalize sensor values to 0-100 range for metrics
  const normalized = sensors.map(s => Math.min(100, Math.max(0, s / 100)));

  // Calculate different aspects of health
  const avgSensors = normalized.reduce((a, b) => a + b, 0) / normalized.length;
  const variance = normalized.reduce((sum, val) => sum + Math.pow(val - avgSensors, 2), 0) / normalized.length;
  const stability = 100 - (variance * 10); // Lower variance = higher stability

  return {
    confidence: Math.min(100, 90 + (cycle % 10)),
    structural_drift: Math.max(0, 30 - (avgSensors * 0.3)),
    relational_stability: Math.max(0, stability),
    coherence: Math.min(100, 85 + (Math.random() * 10)),
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
  if (drift > 50 || stability < 30) return 'critical';
  if (drift > 30 || stability < 50) return 'warning';
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

  return unit1Rows.map((row, index) => {
    const cycle = parseInt(row.cycle, 10);
    const sensors = extractSensorValues(row);
    const metrics = calculateMetrics(sensors, cycle);
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
