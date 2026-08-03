// 프로덕션(Vercel)에서는 추론 서버가 별도로 구동되므로, 빌드 시 환경변수
// VITE_API_BASE에 그 서버의 URL(예: https://your-server.example.com)을
// 설정해야 한다. 로컬 개발에서는 비워두면 vite.config.js의 프록시가
// localhost:8765(server.py)로 요청을 전달한다.
export const API_BASE = import.meta.env.VITE_API_BASE || '';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}
