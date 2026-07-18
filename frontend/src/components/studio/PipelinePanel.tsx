import {useEffect, useState} from 'react';
import {Box, ChevronRight, Play, RotateCcw, ShieldCheck, Zap} from 'lucide-react';
import CharacterViewport from '../../CharacterViewport';
import {api} from '../../api';
import type {Job} from '../../types';

type Props = {
  job: Job;
  running: boolean;
  workerMessage: string;
  artifactUrl: string;
  onRunNext: () => Promise<void>;
  onRunAll: () => Promise<void>;
  onReset: (stageId: string) => Promise<void>;
};

// 파이프라인 실행/재실행 + 최신 아티팩트 프리뷰 + 로그.
export default function PipelinePanel({job, running, workerMessage, artifactUrl, onRunNext, onRunAll, onReset}: Props) {
  const [pendingReset, setPendingReset] = useState<string | null>(null);
  const [thumb, setThumb] = useState('');

  useEffect(() => {
    let alive = true;
    setThumb('');
    if (!job.image) return;
    api.readJobImage(job.id).then(u => { if (alive && u) setThumb(u); }).catch(() => {});
    return () => { alive = false; };
  }, [job.id, job.image]);
  useEffect(() => setPendingReset(null), [job.id]);

  const complete = job.status === 'complete';
  const logs = [...(job.logs || [])].slice(-60).reverse();

  return (
    <>
      <div className="workspace">
        <div className="viewport">
          {artifactUrl ? (
            <>
              <div className="viewtop"><span>LATEST ARTIFACT · DRAG TO ORBIT · SCROLL TO ZOOM</span><div /></div>
              <CharacterViewport playing clip="Idle" speed={1} time={0} showSkeleton={false} modelUrl={artifactUrl} />
              <div className="scene-note"><Box /><div><b>Latest GLB artifact</b><span>Motion 탭에서 clip 재생 · skeleton 확인</span></div></div>
            </>
          ) : (
            <div className="viewport-placeholder">
              {thumb ? <img src={thumb} alt="Reference" /> : <Box />}
              <b>{thumb ? 'Reference image' : 'No preview yet'}</b>
              <span>Run pipeline을 실행하면 3D 아티팩트가 여기에 표시됩니다.</span>
            </div>
          )}
        </div>
        <div className="inspector">
          <div className="ins-head">
            <div><span>PIPELINE</span><h2>{job.name}</h2></div>
          </div>
          {job.imageHash && <div className="provenance">SOURCE VERIFIED <b>{job.imageHash.slice(0, 12)}</b></div>}
          <div className="progress">
            <div><span>{workerMessage || 'Pipeline progress'}</span><b>{job.progress}%</b></div>
            <i><em className={running ? 'indeterminate' : ''} style={{width: `${job.progress}%`}} /></i>
          </div>
          <div className="steps">
            {job.stages.map((s, i) => (
              <div className={`step ${s.status}`} key={s.id}>
                <div className="stepnum">{s.status === 'done' ? '✓' : s.status === 'running' ? '··' : i + 1}</div>
                <div><b>{s.name}</b><span>{s.detail}</span></div>
                {(s.status === 'done' || s.status === 'failed') && !running ? (
                  pendingReset === s.id ? (
                    <span className="reset-confirm">
                      <button className="mini danger" onClick={async () => { setPendingReset(null); await onReset(s.id); }}>Reset</button>
                      <button className="mini" onClick={() => setPendingReset(null)}>×</button>
                    </span>
                  ) : (
                    <button className="mini reset" aria-label={`Reset ${s.name}`} title="이 단계부터 다시 실행" onClick={() => setPendingReset(s.id)}><RotateCcw /></button>
                  )
                ) : <ChevronRight />}
              </div>
            ))}
          </div>
          <button className="run" disabled={running || complete || !job.image} onClick={onRunNext}>
            <Zap />{running ? 'Worker running…' : complete ? 'Pipeline complete' : job.image ? 'Run next stage' : 'Reference image required'}
          </button>
          <button className="import-motion" disabled={running || complete || !job.image} onClick={onRunAll}>
            <Play />Run full pipeline
          </button>
          <p className="hint"><ShieldCheck /> Every stage is non-destructive · 완료 단계는 ↺ 로 재실행</p>
        </div>
      </div>
      <details className="log-panel">
        <summary>Pipeline logs <em>{job.logs?.length || 0}</em></summary>
        <div className="log-list">
          {logs.length ? logs.map((l, i) => (
            <div key={i} className={`log-row ${l.level}`}>
              <span>{new Date(l.time).toLocaleTimeString()}</span>
              <b>{l.stage}</b>
              <p>{l.message}</p>
            </div>
          )) : <div className="log-row"><p>아직 로그가 없습니다.</p></div>}
        </div>
      </details>
    </>
  );
}
