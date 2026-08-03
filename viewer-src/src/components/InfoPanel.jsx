import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { computePixelStats } from '../lib/mockData';

const LABEL = { fontSize: 10, fill: '#aaa' };

export default function InfoPanel({ patient, sliceIndex, sliceData }) {
  const stats = useMemo(() => {
    if (!sliceData) return null;
    return computePixelStats(sliceData.data);
  }, [sliceData]);

  const histData = useMemo(() => {
    if (!stats) return [];
    return stats.histogram.filter((_, i) => i % 2 === 0);
  }, [stats]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#0f0f1f', borderRight: '1px solid #2a2a4a', width: 220 }}>

      {/* Patient Info */}
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #2a2a4a' }}>
        <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', marginBottom: 4 }}>환자 정보</div>
        <div style={row}><span style={lbl}>ID</span><span style={{ ...val, fontSize: 10 }}>{patient.id}</span></div>
        <div style={row}><span style={lbl}>Modality</span><span style={val}>{patient.modality}</span></div>
        <div style={row}><span style={lbl}>총 슬라이스</span><span style={val}>{patient.sliceCount}</span></div>
        <div style={row}><span style={lbl}>두께</span><span style={val}>{patient.sliceThickness} mm</span></div>
      </div>

      {/* Histogram */}
      <div style={{ padding: '10px 8px', borderBottom: '1px solid #2a2a4a', flex: '0 0 auto' }}>
        <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', marginBottom: 6 }}>픽셀 값 분포</div>
        <div style={{ fontSize: 10, color: '#888', marginBottom: 4 }}>범위: 0 ~ 255 (배경 제외)</div>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={histData} margin={{ top: 0, right: 0, left: -28, bottom: 0 }}>
            <XAxis
              dataKey="val"
              tick={LABEL}
              tickFormatter={v => v === 0 || v === 128 || v === 255 ? v : ''}
              interval={9}
            />
            <YAxis tick={LABEL} />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid #ce93d8', fontSize: 11 }}
              formatter={(v, n, p) => [v, `val ${p.payload.val}`]}
              labelFormatter={() => ''}
            />
            <Bar dataKey="count" fill="#ce93d8" opacity={0.8} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Slice stats */}
      <div style={{ padding: '10px 12px', flex: 1 }}>
        <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', marginBottom: 8 }}>슬라이스 정보</div>
        <div style={row}><span style={lbl}>슬라이스 #</span><span style={val}>{sliceIndex + 1} / {patient.sliceCount}</span></div>
        <div style={row}><span style={lbl}>위치</span><span style={val}>{(sliceIndex * patient.sliceThickness).toFixed(1)} mm</span></div>

        {stats && (
          <>
            <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 'bold', margin: '10px 0 6px' }}>픽셀 통계</div>
            <div style={row}><span style={lbl}>최솟값</span><span style={{ ...val, color: '#81d4fa' }}>{stats.min}</span></div>
            <div style={row}><span style={lbl}>최댓값</span><span style={{ ...val, color: '#f48fb1' }}>{stats.max}</span></div>
            <div style={row}><span style={lbl}>평균</span><span style={{ ...val, color: '#a5d6a7' }}>{stats.mean}</span></div>
            <div style={row}><span style={lbl}>표준편차</span><span style={{ ...val, color: '#ffcc80' }}>{stats.std}</span></div>

            <div style={{ marginTop: 12 }}>
              <div style={{ color: '#666', fontSize: 10, marginBottom: 4 }}>MRI T1 참조값</div>
              {[
                { label: '배경',   range: '0',         color: '#444' },
                { label: 'CSF',    range: '20 ~ 80',   color: '#64b5f6' },
                { label: '회백질', range: '80 ~ 140',  color: '#81c784' },
                { label: '백질',   range: '140 ~ 215', color: '#e0e0e0' },
                { label: '고강도', range: '215 ~ 255', color: '#ffcc80' },
              ].map(({ label, range, color }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid #1a1a2e' }}>
                  <span style={{ fontSize: 10, color }}>{label}</span>
                  <span style={{ fontSize: 10, color: '#888', fontFamily: 'monospace' }}>{range}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const row = { display: 'flex', justifyContent: 'space-between', padding: '2px 0', alignItems: 'center' };
const lbl = { fontSize: 11, color: '#888' };
const val = { fontSize: 11, color: '#e0e0e0', fontFamily: 'monospace' };
