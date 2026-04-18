export default function Blob() {
  return (
    <div aria-hidden style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
      <div style={{
        position: 'absolute', top: '-10%', left: '-5%',
        width: 520, height: 520, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(108,143,255,0.13) 0%, transparent 70%)',
        animation: 'blobFloat 9s ease-in-out infinite',
        filter: 'blur(40px)',
      }} />
      <div style={{
        position: 'absolute', bottom: '5%', right: '-8%',
        width: 420, height: 420, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(167,139,250,0.11) 0%, transparent 70%)',
        animation: 'blobFloat 12s ease-in-out infinite reverse',
        filter: 'blur(50px)',
      }} />
      <div style={{
        position: 'absolute', top: '45%', left: '55%',
        width: 300, height: 300, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(74,222,128,0.07) 0%, transparent 70%)',
        animation: 'blobFloat 15s ease-in-out infinite 3s',
        filter: 'blur(35px)',
      }} />
    </div>
  );
}
