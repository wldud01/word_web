export default function GlassCard({ children, style, onClick, hover = true }) {
  return (
    <div
      onClick={onClick}
      style={{
        background: 'rgba(255,255,255,0.06)',
        border: '1px solid rgba(255,255,255,0.10)',
        borderRadius: 24,
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        transition: hover ? 'background 0.2s, border-color 0.2s, transform 0.2s' : undefined,
        cursor: onClick ? 'pointer' : undefined,
        ...style,
      }}
      onMouseEnter={hover && onClick ? e => {
        e.currentTarget.style.background = 'rgba(255,255,255,0.10)';
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.18)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      } : undefined}
      onMouseLeave={hover && onClick ? e => {
        e.currentTarget.style.background = 'rgba(255,255,255,0.06)';
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.10)';
        e.currentTarget.style.transform = 'translateY(0)';
      } : undefined}
    >
      {children}
    </div>
  );
}
