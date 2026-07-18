import {useCallback, useEffect, useMemo, useState} from 'react';
import {Box, Film, Pause, Play, RotateCcw, Upload} from 'lucide-react';
import CharacterViewport, {type PlaybackState} from '../../LazyViewport';

type Props = {
  modelUrl: string;
  usingFallback: boolean;
  onLoadFile: () => void;
};

// 스켈레탈 clip 재생 · 타임라인 스크럽 · 속도/스켈레톤 제어.
export default function MotionPanel({modelUrl, usingFallback, onLoadFile}: Props) {
  const [playing, setPlaying] = useState(true);
  const [clip, setClip] = useState('Idle');
  const [speed, setSpeed] = useState(1);
  const [scrub, setScrub] = useState(0);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [playback, setPlayback] = useState<PlaybackState>({duration: 0, time: 0, clips: [], active: '', loaded: false});

  useEffect(() => { if (playing) setScrub(playback.time); }, [playback.time, playing]);
  const onPlayback = useCallback((s: PlaybackState) => setPlayback(s), []);
  const available = useMemo(() => (playback.clips.length ? playback.clips : ['Idle', 'Walk', 'Run']), [playback.clips]);

  return (
    <>
      <div className="workspace motion-workspace">
        <div className="viewport">
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
            {available.map((name, i) => (
              <button key={name} className={clip === name || playback.active === name ? 'selected' : ''} onClick={() => { setClip(name); setPlaying(true); }}>
                <span className="clip-icon"><Film /></span>
                <span><b>{name}</b><small>{i === 0 ? 'Loop · locomotion' : 'Embedded skeletal clip'}</small></span>
                <em>{playback.active === name ? 'LIVE' : 'PLAY'}</em>
              </button>
            ))}
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
        <button className="transport" aria-label={playing ? 'Pause' : 'Play'} onClick={() => setPlaying(x => !x)}>{playing ? <Pause /> : <Play />}</button>
        <span>{scrub.toFixed(2)}s</span>
        <input type="range" aria-label="Timeline scrub" min="0" max={playback.duration || 1} step=".01"
          value={Math.min(scrub, playback.duration || 1)} onChange={e => { setPlaying(false); setScrub(+e.target.value); }} />
        <span>{(playback.duration || 0).toFixed(2)}s</span>
      </div>
    </>
  );
}
