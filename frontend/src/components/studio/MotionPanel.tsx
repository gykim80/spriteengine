import {useCallback, useEffect, useState} from 'react';
import {Box, Film, Pause, Play, RotateCcw, Upload} from 'lucide-react';
import CharacterViewport, {type PlaybackState} from '../../LazyViewport';

type Props = {
  modelUrl: string;
  usingFallback: boolean;
  onLoadFile: () => void;
  onPreviewFile: (f: File) => void;
  setNotice: (s: string) => void;
};

// 스켈레탈 clip 재생 · 타임라인 스크럽 · 속도/스켈레톤 제어.
export default function MotionPanel({modelUrl, usingFallback, onLoadFile, onPreviewFile, setNotice}: Props) {
  const [playing, setPlaying] = useState(true);
  const [clip, setClip] = useState('');
  const [speed, setSpeed] = useState(1);
  const [scrub, setScrub] = useState(0);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [playback, setPlayback] = useState<PlaybackState>({duration: 0, time: 0, clips: [], active: '', loaded: false});
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => { if (playing) setScrub(playback.time); }, [playback.time, playing]);
  // 모델이 바뀌면 이전 모델의 clip 선택/scrub을 초기화
  useEffect(() => { setClip(''); setScrub(0); setPlaying(true); }, [modelUrl]);
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
          <CharacterViewport playing={playing} clip={clip} speed={speed} time={scrub} showSkeleton={showSkeleton} modelUrl={modelUrl} onState={onPlayback} />
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
          <div className="clip-list">
            {playback.clips.map(name => (
              <button key={name} className={clip === name || playback.active === name ? 'selected' : ''} onClick={() => { setClip(name); setPlaying(true); }}>
                <span className="clip-icon"><Film /></span>
                <span><b>{name}</b><small>Embedded skeletal clip</small></span>
                <em>{playback.active === name ? 'LIVE' : 'PLAY'}</em>
              </button>
            ))}
            {!playback.clips.length && (
              <div className="clip-empty">
                {playback.loaded
                  ? '이 GLB에는 embedded animation clip이 없습니다. Import GLB로 애니메이션 포함 파일을 로드하세요.'
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
