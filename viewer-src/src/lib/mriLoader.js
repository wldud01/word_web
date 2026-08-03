import { apiUrl } from './api';

// { data: Int16Array, width, height } → PNG Blob (서버 POST용)
export function sliceToBlob(slice) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width  = slice.width;
    canvas.height = slice.height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(slice.width, slice.height);
    for (let i = 0; i < slice.data.length; i++) {
      const v = Math.max(0, Math.min(255, slice.data[i]));
      img.data[i * 4]     = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    canvas.toBlob(resolve, 'image/png');
  });
}

// Decode a PNG Blob/ArrayBuffer → { data: Int16Array, width, height }
export async function loadSliceFromBlob(blob) {
  const url = URL.createObjectURL(blob instanceof Blob ? blob : new Blob([blob]));
  try {
    return await loadMRISlice(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Load a PNG from URL and return { data: Int16Array, width, height }
export async function loadMRISlice(url) {
  const img = await new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`이미지 로드 실패: ${url}`));
    image.src = url;
  });

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const rgba = ctx.getImageData(0, 0, img.width, img.height).data;
  const data = new Int16Array(img.width * img.height);
  for (let i = 0; i < data.length; i++) {
    data[i] = rgba[i * 4]; // R channel (grayscale PNG: R=G=B)
  }

  return { data, width: img.width, height: img.height };
}

// Load all MRI slices from server with progress callback
// onProgress(done, total, partialSlices)
export async function loadAllMRISlices(onProgress) {
  const res = await fetch(apiUrl('/api/mri-slices'));
  if (!res.ok) throw new Error('MRI 슬라이스 목록을 불러올 수 없어요 (추론 서버가 실행 중인지 확인)');
  const { files } = await res.json();

  const slices = [];
  for (let i = 0; i < files.length; i++) {
    const slice = await loadMRISlice(apiUrl(`/mri/${files[i]}`));
    slices.push(slice);
    onProgress?.(i + 1, files.length, slices);
  }
  return slices;
}
