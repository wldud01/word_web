// Renders a CT slice (Int16Array) onto a canvas using windowing (WC/WW)
export function renderSliceToCanvas(canvas, pixelData, width, height, wc, ww) {
  const ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;

  const imageData = ctx.createImageData(width, height);
  const buf = imageData.data;

  const lower = wc - ww / 2;
  const upper = wc + ww / 2;

  for (let i = 0; i < pixelData.length; i++) {
    let v = pixelData[i];
    // Window clamp
    let gray;
    if (v <= lower) gray = 0;
    else if (v >= upper) gray = 255;
    else gray = Math.round(((v - lower) / ww) * 255);

    buf[i * 4] = gray;
    buf[i * 4 + 1] = gray;
    buf[i * 4 + 2] = gray;
    buf[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

// Renders a small thumbnail (downsampled)
export function renderThumbnail(canvas, pixelData, width, height, wc = 128, ww = 256) {
  const thumbSize = 80;
  canvas.width = thumbSize;
  canvas.height = thumbSize;
  const ctx = canvas.getContext('2d');

  // Offscreen full-size
  const offscreen = document.createElement('canvas');
  offscreen.width = width;
  offscreen.height = height;
  renderSliceToCanvas(offscreen, pixelData, width, height, wc, ww);

  ctx.drawImage(offscreen, 0, 0, thumbSize, thumbSize);
}

// Overlay AI inference result (colored highlight) on contrast image
export function renderInferenceOverlay(canvas, pixelData, width, height, wc, ww) {
  renderSliceToCanvas(canvas, pixelData, width, height, wc, ww);
  const ctx = canvas.getContext('2d');

  // Simulate AI detection: highlight region with semi-transparent color
  ctx.save();
  ctx.globalAlpha = 0.35;

  // Simulate nodule detection in right lung
  ctx.fillStyle = '#ff4444';
  ctx.beginPath();
  ctx.ellipse(width * 0.62, height * 0.42, 18, 14, 0.3, 0, Math.PI * 2);
  ctx.fill();

  // Simulate consolidation area
  ctx.fillStyle = '#ffaa00';
  ctx.beginPath();
  ctx.ellipse(width * 0.38, height * 0.48, 28, 22, -0.2, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = 1;

  // Labels
  ctx.fillStyle = '#ff4444';
  ctx.font = 'bold 11px sans-serif';
  ctx.fillText('결절 의심', width * 0.63 + 20, height * 0.41);

  ctx.fillStyle = '#ffaa00';
  ctx.fillText('간유리음영', width * 0.25, height * 0.45 - 26);

  ctx.restore();
}
