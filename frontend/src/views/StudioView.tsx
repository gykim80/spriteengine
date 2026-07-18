import {useEffect, useMemo, useRef, useState} from 'react';
import {Clapperboard} from 'lucide-react';
import {api, errText} from '../api';
import type {Job, StudioTab} from '../types';
import PipelinePanel from '../components/studio/PipelinePanel';
import MotionPanel from '../components/studio/MotionPanel';
import ExportPanel from '../components/studio/ExportPanel';

type Props = {
  job: Job | null;
  running: boolean;
  workerMessage: string;
  onRunNext: () => Promise<void>;
  onRunAll: () => Promise<void>;
  onReset: (stageId: string) => Promise<void>;
  onExport: () => Promise<void>;
  goProjects: () => void;
  setNotice: (s: string) => void;
};

// 선택된 프로젝트의 통합 편집 화면: Pipeline · Motion · Export 탭.
export default function StudioView({job, running, workerMessage, onRunNext, onRunAll, onReset, onExport, goProjects, setNotice}: Props) {
  const [tab, setTab] = useState<StudioTab>('pipeline');
  const [artifactUrl, setArtifactUrl] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [customModel, setCustomModel] = useState('');
  const blobRef = useRef('');

  const latestGlb = useMemo(
    () => [...(job?.artifacts || [])].reverse().find(a => a.path.toLowerCase().endsWith('.glb')),
    [job?.artifacts],
  );

  // 최신 GLB 아티팩트를 뷰포트용 data URI로 로드
  useEffect(() => {
    let alive = true;
    if (!latestGlb) { setArtifactUrl(''); return; }
    api.readArtifact(latestGlb.path)
      .then(u => { if (alive) setArtifactUrl(u || ''); })
      .catch(() => { if (alive) setArtifactUrl(''); });
    return () => { alive = false; };
  }, [latestGlb?.path]);

  // 프로젝트가 바뀌면 프리뷰/커스텀 모델 초기화
  useEffect(() => {
    setPreviewUrl('');
    if (blobRef.current) { URL.revokeObjectURL(blobRef.current); blobRef.current = ''; }
    setCustomModel('');
    setTab('pipeline');
  }, [job?.id]);
  useEffect(() => () => { if (blobRef.current) URL.revokeObjectURL(blobRef.current); }, []);

  // 로컬 GLB/glTF를 blob URL로 Motion 뷰포트에 로드 (파일 선택·drag & drop 공용)
  function previewModelFile(f: File) {
    if (blobRef.current) URL.revokeObjectURL(blobRef.current);
    blobRef.current = URL.createObjectURL(f);
    setCustomModel(blobRef.current);
    setNotice(`${f.name} loaded locally. Embedded clips will appear in Motion.`);
  }
  function loadModelFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.glb,.gltf';
    input.onchange = () => {
      const f = input.files?.[0];
      if (f) previewModelFile(f);
    };
    input.click();
  }
  async function previewArtifact(path: string) {
    try {
      const u = await api.readArtifact(path);
      if (u) { setPreviewUrl(u); setTab('motion'); setNotice('Artifact가 Motion 탭에 로드되었습니다.'); }
    } catch (e) {
      setNotice(errText(e));
    }
  }

  if (!job) {
    return (
      <section className="content">
        <div className="studio-empty">
          <Clapperboard />
          <h2>선택된 프로젝트가 없습니다</h2>
          <p>Projects에서 프로젝트를 열거나 새 reference image를 import하면 Studio가 활성화됩니다.</p>
          <button className="run" onClick={goProjects}>Go to Projects</button>
        </div>
      </section>
    );
  }

  const motionModel = customModel || previewUrl || artifactUrl;
  const tabOrder: StudioTab[] = ['pipeline', 'motion', 'export'];
  // WAI-ARIA tabs 패턴: ←/→로 탭 이동, 선택된 탭만 tab 순서에 노출 (roving tabindex)
  function onTabsKey(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const i = tabOrder.indexOf(tab);
    const next = tabOrder[(i + (e.key === 'ArrowRight' ? 1 : tabOrder.length - 1)) % tabOrder.length];
    setTab(next);
    (e.currentTarget.querySelector(`[data-tab="${next}"]`) as HTMLElement | null)?.focus();
  }
  return (
    <section className="content">
      <div className="toolbar">
        <div className="tabs studio-tabs" role="tablist" onKeyDown={onTabsKey}>
          {([['pipeline', 'Pipeline'], ['motion', 'Motion'], ['export', 'Export']] as Array<[StudioTab, string]>).map(([t, label]) => (
            <button key={t} role="tab" data-tab={t} id={`studio-tab-${t}`} aria-controls="studio-panel"
              aria-selected={tab === t} tabIndex={tab === t ? 0 : -1}
              className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{label}</button>
          ))}
        </div>
        <div className={`status ${running ? 'pending' : ''}`}>
          <i /> {running ? (workerMessage || 'Worker running') : `${job.status.toUpperCase()} · ${job.progress}%`}
          <span>{job.id}</span>
        </div>
      </div>
      <div role="tabpanel" id="studio-panel" aria-labelledby={`studio-tab-${tab}`}>
        {tab === 'pipeline' && (
          <PipelinePanel job={job} running={running} workerMessage={workerMessage} artifactUrl={artifactUrl}
            onRunNext={onRunNext} onRunAll={onRunAll} onReset={onReset} />
        )}
        {tab === 'motion' && (
          <MotionPanel modelUrl={motionModel || '/models/Soldier.glb'} usingFallback={!motionModel}
            onLoadFile={loadModelFile} onPreviewFile={previewModelFile} setNotice={setNotice} />
        )}
        {tab === 'export' && (
          <ExportPanel job={job} running={running} onExport={onExport} onPreview={previewArtifact} />
        )}
      </div>
    </section>
  );
}
