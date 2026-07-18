import {lazy, Suspense, type ComponentProps} from 'react';

// Three.js가 포함된 CharacterViewport를 별도 청크로 분리해 초기 번들을 줄인다.
const Viewport = lazy(() => import('./CharacterViewport'));

export type {PlaybackState} from './CharacterViewport';

export default function LazyViewport(props: ComponentProps<typeof Viewport>) {
  return (
    <Suspense fallback={<div className="viewport-loading">3D viewport 로딩 중…</div>}>
      <Viewport {...props} />
    </Suspense>
  );
}
