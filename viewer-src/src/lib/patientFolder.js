// 환자 폴더 업로드 → 파일명(또는 경로)에 t1/t2가 포함된 파일을 각각의
// 모달리티로 분류한다. "t1ce" 같은 다른 모달리티가 t1으로 잘못 분류되지
// 않도록 단어 경계를 확인한다.
const IMAGE_EXT = /\.(png|jpe?g)$/i;
const T1_RE = /(^|[^a-z0-9])t1([^a-z0-9]|$)/;
const T2_RE = /(^|[^a-z0-9])t2([^a-z0-9]|$)/;

export function classifyByModality(files) {
  const t1 = [];
  const t2 = [];
  for (const f of files) {
    if (!IMAGE_EXT.test(f.name)) continue;
    const key = (f.webkitRelativePath || f.name).toLowerCase();
    if (T1_RE.test(key)) t1.push(f);
    else if (T2_RE.test(key)) t2.push(f);
  }
  const byName = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true });
  t1.sort(byName);
  t2.sort(byName);
  return { t1, t2 };
}

async function readAllEntries(dirReader) {
  const all = [];
  while (true) {
    const batch = await new Promise((res) => dirReader.readEntries(res));
    if (!batch.length) break;
    all.push(...batch);
  }
  return all;
}

async function collectFiles(entry, out) {
  if (entry.isFile) {
    const f = await new Promise((res) => entry.file(res));
    out.push(f);
  } else if (entry.isDirectory) {
    const children = await readAllEntries(entry.createReader());
    for (const child of children) await collectFiles(child, out);
  }
}

// 드래그&드롭 DataTransferItemList → File[]
export async function filesFromDataTransferItems(items) {
  const out = [];
  for (const item of items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) await collectFiles(entry, out);
  }
  return out;
}

// <input webkitdirectory> FileList → File[]
export function filesFromFileList(fileList) {
  return [...fileList];
}

export function folderNameFromFiles(files) {
  const rel = files.find(f => f.webkitRelativePath)?.webkitRelativePath;
  return rel ? rel.split('/')[0] : '';
}
