import {useCallback, useEffect, useState} from 'react';
import {Box, Download, Film, Pause, Play, RotateCcw, Sparkles, Upload, Wand2} from 'lucide-react';
import CharacterViewport, {type PlaybackState} from '../../LazyViewport';
import {parseMotionPrompt, type MotionSpec} from '../../motion/motionScript';
import {api, errText, isCancelled} from '../../api';

type Props = {
  modelUrl: string;
  usingFallback: boolean;
  jobName: string;
  jobId: string;
  onPreviewArtifact: (path: string) => Promise<void>;
  onLoadFile: () => void;
  onPreviewFile: (f: File) => void;
  setNotice: (s: string) => void;
};

// 스켈레탈 clip 재생 · 타임라인 스크럽 · 속도/스켈레톤 제어.
export default function MotionPanel({modelUrl, usingFallback, jobName, jobId, onPreviewArtifact, onLoadFile, onPreviewFile, setNotice}: Props) {
  const [playing, setPlaying] = useState(true);
  const [clip, setClip] = useState('');
  const [speed, setSpeed] = useState(1);
  const [scrub, setScrub] = useState(0);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [playback, setPlayback] = useState<PlaybackState>({duration: 0, time: 0, clips: [], active: '', loaded: false});
  const [dragOver, setDragOver] = useState(false);
  // 텍스트 연출: 프롬프트 → 파싱된 spec → viewport에서 AnimationClip으로 컴파일
  const [prompt, setPrompt] = useState('');
  const [motion, setMotion] = useState<MotionSpec | null>(null);
  const [exporting, setExporting] = useState(false);
  const [generating, setGenerating] = useState(false);

  useEffect(() => { if (playing) setScrub(playback.time); }, [playback.time, playing]);
  // 모델이 바뀌면 이전 모델의 clip 선택/scrub/연출을 초기화
  useEffect(() => { setClip(''); setScrub(0); setPlaying(true); setMotion(null); }, [modelUrl]);
  const onPlayback = useCallback((s: PlaybackState) => setPlayback(s), []);

  // 스튜디오 툴 표준 단축키: Space = 재생/일시정지, ←/→ = 0.1s 스크럽 (폼 요소 포커스 시 제외)
  useEffect(() => {
    const duration = playback.duration;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input,textarea,select,button,[contenteditable]')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying(x => !x);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const step = e.key === 'ArrowRight' ? 0.1 : -0.1;
        setPlaying(false);
        setScrub(s => Math.min(Math.max(0, s + step), duration || 0));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [playback.duration]);

  // Library·Projects와 동일한 UX: GLB/glTF를 viewport에 드롭하면 즉시 미리보기
  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = Array.from(e.dataTransfer.files).find(x => /\.(glb|gltf)$/i.test(x.name));
    if (!f) {
      setNotice('GLB 또는 glTF 파일만 드롭할 수 있습니다.');
      return;
    }
    onPreviewFile(f);
  }

  // "날아서 발차기하는 애니메이션" 같은 문장을 파싱해 연출 clip을 만들어 즉시 재생
  function directMotion() {
    const spec = parseMotionPrompt(prompt);
    if (!spec) {
      setNotice('인식할 수 있는 동작이 없습니다. 예: "날아서 발차기하는 애니메이션", "점프하고 회전"');
      return;
    }
    setMotion(spec);
    setClip(spec.clipName);
    setScrub(0);
    setPlaying(true);
    const seq = spec.actions.map(a => (a.repeat > 1 ? `${a.label} ×${a.repeat}` : a.label)).join(' → ');
    setNotice(`연출 적용: ${seq}${spec.tempoLabel ? ` · ${spec.tempoLabel}` : ''}`);
  }

  // HY-Motion(RunPod)으로 자연어 → 실제 스켈레탈 모션을 생성해 rigged GLB에 베이킹.
  // 키워드 연출(directMotion)과 달리 임의 문장을 이해한다. GPU cold start 시 수 분 소요.
  async function generateAIMotion() {
    const text = prompt.trim();
    if (!text || generating) return;
    setGenerating(true);
    setNotice('HY-Motion으로 모션 생성 중… (GPU cold start 시 몇 분 걸릴 수 있습니다)');
    try {
      const result = await api.runPodGenerateMotion(jobId, [{id: 'motion1', text, duration: 5}]);
      const failed = Object.entries(result.errors || {});
      await onPreviewArtifact(result.path);
      setNotice(`AI 모션 ${result.clips}개 클립 베이킹 완료 (${result.model})${failed.length ? ` · 실패: ${failed.map(([k, v]) => `${k}: ${v}`).join(', ')}` : ''}`);
    } catch (e) {
      if (!isCancelled(e)) setNotice(errText(e));
    } finally {
      setGenerating(false);
    }
  }

  function clearMotion() {
    if (motion && (clip === motion.clipName)) setClip('');
    setMotion(null);
    setNotice('연출을 해제했습니다.');
  }

  // 연출 clip을 모델에 bake한 GLB를 저장한다 (embedded clip 유지).
  // three가 필요한 bake 모듈은 dynamic import로 로드해 메인 번들을 지킨다.
  async function exportAnimated() {
    if (!motion || exporting) return;
    setExporting(true);
    try {
      const {bakeAnimatedGLBBase64} = await import('../../motion/exportAnimated');
      const b64 = await bakeAnimatedGLBBase64(modelUrl, motion);
      const dst = await api.saveAnimatedGLB(jobName, b64);
      setNotice(`애니메이션 포함 GLB 저장 완료: ${dst}`);
    } catch (e) {
      if (!isCancelled(e)) setNotice(errText(e));
    } finally {
      setExporting(false);
    }
  }

  return (
    <>
      <div className="workspace motion-workspace">
        <div className={`viewport ${dragOver ? 'drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={e => { if (e.currentTarget === e.target) setDragOver(false); }}
          onDrop={handleDrop}>
          <div className="viewtop">
            <span>SKINNED MESH · DRAG TO ORBIT · SCROLL TO ZOOM</span>
            <div>
              <button onClick={() => setShowSkeleton(x => !x)}>{showSkeleton ? 'Hide' : 'Show'} skeleton</button>
              <button onClick={onLoadFile}>Load GLB</button>
            </div>
          </div>
          <CharacterViewport playing={playing} clip={clip} speed={speed} time={scrub} showSkeleton={showSkeleton} modelUrl={modelUrl} motion={motion} onState={onPlayback} />
          <div className="axis"><b>Y</b><span>X</span><em>Z</em></div>
          <div className="scene-note">
            <Box />
            <div>
              <b>{usingFallback ? 'Bundled demo character' : playback.loaded ? 'Project artifact' : 'Loading asset…'}</b>
              <span>{usingFallback ? '아직 프로젝트 GLB가 없어 데모 캐릭터를 표시합니다' : `${playback.clips.length} embedded animation clips · crossfade`}</span>
            </div>
          </div>
        </div>
        <div className="inspector motion-panel">
          <div className="ins-head">
            <div><span>MOTION CLIPS</span><h2>Animation browser</h2></div>
            <b className="fps">60 FPS</b>
          </div>
          <div className="motion-director">
            <label htmlFor="motion-prompt">텍스트로 애니메이션 연출</label>
            <div className="director-row">
              <input id="motion-prompt" type="text" placeholder='예: 날아서 발차기하는 애니메이션'
                value={prompt} onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') directMotion(); }} />
              <button onClick={directMotion} disabled={!prompt.trim()} title="프롬프트에서 동작을 추출해 연출 clip을 생성합니다"><Wand2 />연출</button>
            </div>
            <button className="director-export" onClick={generateAIMotion} disabled={!prompt.trim() || generating}
              title="HY-Motion-1.0(RunPod GPU)으로 자연어에서 실제 모션을 생성해 리깅된 모델에 베이킹합니다">
              <Sparkles />{generating ? 'AI 모션 생성 중…' : 'AI 모션 생성 (HY-Motion)'}
            </button>
            {motion && (
              <>
                <div className="director-chips" aria-label="연출된 동작 시퀀스">
                  {motion.actions.map((a, i) => (
                    <span key={a.id} className="chip">{i > 0 ? '→ ' : ''}{a.label}{a.repeat > 1 ? ` ×${a.repeat}` : ''}</span>
                  ))}
                  {motion.tempoLabel && <span className="chip chip-tempo">{motion.tempoLabel}</span>}
                  <button className="chip-clear" onClick={clearMotion}>해제</button>
                </div>
                <button className="director-export" onClick={exportAnimated} disabled={exporting}
                  title="연출 애니메이션을 모델에 bake한 GLB를 저장합니다">
                  <Download />{exporting ? 'Baking animation…' : '애니메이션 포함 GLB 저장'}
                </button>
              </>
            )}
          </div>
          <div className="clip-list">
            {playback.clips.map(name => (
              <button key={name} className={clip === name || playback.active === name ? 'selected' : ''} onClick={() => { setClip(name); setPlaying(true); }}>
                <span className="clip-icon"><Film /></span>
                <span><b>{name}</b><small>{motion && name === motion.clipName ? 'Directed motion (text prompt)' : 'Embedded skeletal clip'}</small></span>
                <em>{playback.active === name ? 'LIVE' : 'PLAY'}</em>
              </button>
            ))}
            {!playback.clips.length && (
              <div className="clip-empty">
                {playback.loaded
                  ? '이 GLB에는 embedded animation clip이 없습니다. 위 텍스트 연출로 애니메이션을 만들거나, Import GLB로 clip 포함 파일을 로드하세요.'
                  : playback.error || '모델 로딩 중…'}
              </div>
            )}
          </div>
          <div className="motion-settings">
            <label>Playback speed <b>{speed.toFixed(2)}×</b></label>
            <input type="range" aria-label="Playback speed" min=".25" max="2" step=".05" value={speed} onChange={e => setSpeed(+e.target.value)} />
            <label><input type="checkbox" checked={showSkeleton} onChange={e => setShowSkeleton(e.target.checked)} /> Skeleton overlay</label>
            <button className="import-motion" onClick={onLoadFile}><Upload />Import GLB with animations</button>
          </div>
        </div>
      </div>
      <div className="timeline">
        <button aria-label="Reset playback" onClick={() => { setPlaying(false); setScrub(0); }}><RotateCcw /></button>
        <button className="transport" aria-label={playing ? 'Pause' : 'Play'} title={`${playing ? 'Pause' : 'Play'} (Space)`} onClick={() => setPlaying(x => !x)}>{playing ? <Pause /> : <Play />}</button>
        <span>{scrub.toFixed(2)}s</span>
        <input type="range" aria-label="Timeline scrub" title="←/→ 키로 0.1s 스크럽" min="0" max={playback.duration || 1} step=".01"
          value={Math.min(scrub, playback.duration || 1)} onChange={e => { setPlaying(false); setScrub(+e.target.value); }} />
        <span>{(playback.duration || 0).toFixed(2)}s</span>
      </div>
    </>
  );
}
