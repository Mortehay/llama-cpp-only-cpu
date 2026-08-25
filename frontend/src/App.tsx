import { useState } from 'react'
import { api } from './api'
import { useAsync } from './hooks'
import CoreGenerator from './tabs/CoreGenerator'
import SheetGenerator from './tabs/SheetGenerator'
import ReferenceTab from './tabs/ReferenceTab'
import Gallery from './tabs/Gallery'
import Settings from './tabs/Settings'

const TABS = [
  { id: 'core', label: 'Core Generator' },
  { id: 'sheet', label: 'Spritesheet' },
  { id: 'ref-core', label: 'Reference · Core' },
  { id: 'ref-sprite', label: 'Reference · Sprite' },
  { id: 'ref-tile', label: 'Reference · Tile' },
  { id: 'gallery', label: 'Gallery' },
  { id: 'settings', label: 'Settings & API' },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [tab, setTab] = useState<TabId>('core')
  const mode = useAsync(() => api.authMode(), [])

  return (
    <>
      <nav className="nav">
        <span className="brand">🎮 Pixel Art Generator</span>
        {mode.data && (
          <span
            className={`mode ${mode.data.enforced ? 'locked' : 'open'}`}
            title={mode.data.message}
          >
            {mode.data.enforced ? '🔒 API secured' : '⚠ API open'}
          </span>
        )}
      </nav>

      <div className="wrap">
        <h1>Pixel Art Generator</h1>
        <p className="sub">
          Generate a core character, build spritesheets from it, and teach the
          pipeline your game's palette, grid and camera from reference art.
        </p>

        {/* Tabs are buttons, and visibility is driven purely by which one is
            mounted. The old UI toggled a class against an inline
            `display:none`, which the class could never beat - the Settings tab
            was permanently blank as a result. */}
        <div className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              className={`tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === 'core' && <CoreGenerator />}
        {tab === 'sheet' && <SheetGenerator />}
        {tab === 'ref-core' && <ReferenceTab kind="core" />}
        {tab === 'ref-sprite' && <ReferenceTab kind="sprite" />}
        {tab === 'ref-tile' && <ReferenceTab kind="tile" />}
        {tab === 'gallery' && <Gallery />}
        {tab === 'settings' && <Settings onModeChange={mode.reload} />}
      </div>
    </>
  )
}
