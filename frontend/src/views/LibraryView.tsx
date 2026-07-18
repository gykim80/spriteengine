import {useEffect, useRef, useState} from 'react';
import {Box, ChevronRight, Film, Library, Upload} from 'lucide-react';
import CharacterViewport, {PlaybackState} from '../CharacterViewport';
import {api} from '../api';

const sources = [
  {name: 'Itch.io 3D Characters', detail: 'Creator-licensed character packs', url: 'https://itch.io/game-assets/tag-3d/tag-characters'},
  {name: 'Itch.io Animations', detail: 'Rigged motion asset packs', url: 'https://itch.io/game-assets/tag-3d/tag-animation'},
  {name: 'Quaternius', detail: 'CC0 animated character packs', url: 'https://quaternius.com/'},
  {name: 'Mixamo', detail: 'Auto-rig and motion library', url: 'https://www.mixamo.com/'},
  {name: 'ActorCore Free', detail: 'Mocap motion collection', url: 'https://www.reallusion.com/actorcore/free-motion.html'},
];

type Props = {setNotice: (s: string) => void};

// 로컬 GLB 미리보기 + 라이선스 인지형 외부 에셋 소스.
export default function LibraryView({setNotice}: Props) {
  const [model, setModel] = useState('/models/Soldier.glb');
  const [modelName, setModelName] = useState('Soldier.glb (bundled demo)');
  const [clip, setClip] = useState('');
  const [playing, setPlaying] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [playback, setPlayback] = useState<PlaybackState>({duration: 0, time: 0, clips: [], active: '', loaded: false});
  const blobRef = useRef('');

  // 언마운트 시 blob URL 정리
  useEffect(() => () => {
    if (blobRef.current) URL.revokeObjectURL(blobRef.current);
  }, []);

  function loadGLB() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.glb,.gltf';
    input.onchange = () => {
      const f = input.files?.[0];
      if (!f) return;
      if (blobRef.current) URL.revokeObjectURL(blobRef.current);
      blobRef.current = URL.createObjectURL(f);
      setModel(blobRef.current);
      setModelName(f.name);
      setClip('');
      setPlaying(true);
      setNotice(`${f.name} 로컬 로드 완료. 포함된 clip이 아래에 표시됩니다.`);
    };
    input.click();
  }
  function openURL(url: string) {
    api.openExternal(url).catch(() => window.open(url, '_blank'));
  }

  return (
    <section className="content">
      <div className="workspace library-workspace">
        <div className="viewport">
          <div className="viewtop">
            <span>LIBRARY PREVIEW · DRAG TO ORBIT · SCROLL TO ZOOM</span>
            <div>
              <button onClick={() => setShowSkeleton(x => !x)}>{showSkeleton ? 'Hide' : 'Show'} skeleton</button>
              <button onClick={loadGLB}>Load GLB</button>
            </div>
          </div>
          <CharacterViewport playing={playing} clip={clip} speed={1} time={0} showSkeleton={showSkeleton} modelUrl={model} onState={setPlayback} />
          <div className="scene-note">
            <Box />
            <div>
              <b>{modelName}</b>
              <span>{playback.clips.length} embedded animation clips</span>
            </div>
          </div>
        </div>
        <div className="inspector motion-panel">
          <div className="ins-head">
            <div><span>LOCAL ASSET</span><h2>Clip browser</h2></div>
          </div>
          <div className="clip-list">
            {(playback.clips.length ? playback.clips : ['Idle', 'Walk', 'Run']).map(name => (
              <button key={name} className={playback.active === name ? 'selected' : ''} onClick={() => { setClip(name); setPlaying(true); }}>
                <span className="clip-icon"><Film /></span>
                <span><b>{name}</b><small>Embedded skeletal clip</small></span>
                <em>{playback.active === name ? 'LIVE' : 'PLAY'}</em>
              </button>
            ))}
          </div>
          <div className="motion-settings">
            <label><input type="checkbox" checked={playing} onChange={e => setPlaying(e.target.checked)} /> Playing</label>
            <button className="import-motion" onClick={loadGLB}><Upload />Load local GLB / glTF</button>
          </div>
        </div>
      </div>
      <div className="library-section">
        <div className="asset-head">
          <div><span>LICENSE-AWARE ASSET SOURCES</span><b>Characters & motion libraries</b></div>
          <small>Assets are opened at source; verify each creator license before commercial use.</small>
        </div>
        <div className="library-cards">
          {sources.map(x => (
            <button key={x.name} onClick={() => openURL(x.url)}>
              <Library />
              <span><b>{x.name}</b><small>{x.detail}</small></span>
              <ChevronRight />
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
