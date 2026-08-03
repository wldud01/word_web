export const MRI_PATIENT = {
  id: 'p1 (Brats18_CBICA_AAM_1)',
  modality: 'T1 → T2 (placeholder)',
  series: 1,
  sliceCount: 140,
  pixelSpacing: 1.0,
  sliceThickness: 1.0,
};

export function computePixelStats(data) {
  let min = Infinity, max = -Infinity, sum = 0, count = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v <= 0) continue; // skip background
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    count++;
  }

  const mean = count > 0 ? sum / count : 0;
  let variance = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v <= 0) continue;
    variance += (v - mean) ** 2;
  }
  const std = count > 0 ? Math.sqrt(variance / count) : 0;

  return {
    min: Math.round(min === Infinity ? 0 : min),
    max: Math.round(max === -Infinity ? 0 : max),
    mean: Math.round(mean),
    std: Math.round(std),
  };
}
