import {Download, Eye, FileBox} from 'lucide-react';
import type {Job} from '../../types';

type Props = {
  job: Job;
  running: boolean;
  onExport: () => Promise<void>;
  onPreview: (path: string) => Promise<void>;
};

const base = (p: string) => p.split(/[\\/]/).pop() || p;

// 최종 GLB 저장 + 스테이지별 아티팩트 목록.
export default function ExportPanel({job, running, onExport, onPreview}: Props) {
  const artifacts = job.artifacts || [];
  const hasGLB = artifacts.some(a => a.path.toLowerCase().endsWith('.glb'));
  return (
    <div className="export-grid">
      <div className="export-card">
        <span className="eyebrow">PACKAGE</span>
        <h2>Export character</h2>
        <p>가장 최근 GLB 아티팩트를 검증 후 원하는 위치로 저장합니다.</p>
        <div className="export-formats">
          <button className="run" disabled={running || !hasGLB} onClick={onExport}>
            <Download />{hasGLB ? 'Export GLB…' : 'GLB artifact 없음 · pipeline 먼저 실행'}
          </button>
          <button className="import-motion" disabled title="Blender 연동 필요 · Coming soon">FBX · Coming soon</button>
          <button className="import-motion" disabled title="Blender 연동 필요 · Coming soon">USDZ · Coming soon</button>
        </div>
      </div>
      <div className="export-card">
        <span className="eyebrow">ARTIFACTS</span>
        <h2>Stage outputs</h2>
        {artifacts.length ? (
          <div className="export-list">
            {artifacts.map((a, i) => (
              <div key={i} className="export-row">
                <FileBox />
                <div>
                  <b>{base(a.path)}</b>
                  <small>{a.stage} · {a.kind}{a.metrics?.adapter ? ` · ${String(a.metrics.adapter)}` : ''}</small>
                </div>
                {a.path.toLowerCase().endsWith('.glb') ? (
                  <button className="mini" aria-label={`Preview ${base(a.path)}`} onClick={() => onPreview(a.path)}><Eye />Preview</button>
                ) : <span className="mini-note">{a.path.toLowerCase().endsWith('.json') ? 'metadata' : 'file'}</span>}
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-note">아직 아티팩트가 없습니다. Pipeline 탭에서 단계를 실행하세요.</p>
        )}
      </div>
    </div>
  );
}
