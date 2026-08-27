import { useEffect, useState } from 'react'
import { api } from './api'
import { useAsync } from './hooks'
import CoreGenerator from './tabs/CoreGenerator'
import SheetGenerator from './tabs/SheetGenerator'
import ReferenceTab from './tabs/ReferenceTab'
import Gallery from './tabs/Gallery'
import Settings from './tabs/Settings'
import Training from './tabs/Training'
import Tiles from './tabs/Tiles'
import Maps from './tabs/Maps'
import Worlds from './tabs/Worlds'
import Commands from './tabs/Commands'

const TABS = [
  { id: 'core', label: 'Entity Generation' },
  { id: 'sheet', label: 'Spritesheet' },
  { id: 'ref-core', label: 'Reference · Core' },
  { id: 'ref-sprite', label: 'Reference · Sprite' },
  { id: 'ref-tile', label: 'Reference · Tile' },
  { id: 'ref-map', label: 'Reference · Map' },
  { id: 'tiles', label: 'Tiles' },
  { id: 'maps', label: 'Maps' },
  { id: 'worlds', label: 'Worlds' },
  { id: 'training', label: 'Training' },
  { id: 'gallery', label: 'Gallery' },
  { id: 'commands', label: 'Commands' },
  { id: 'settings', label: 'Settings & API' },
] as const

type TabId = (typeof TABS)[number]['id']

const IDS = TABS.map((t) => t.id) as readonly string[]

/** The tab named in the URL hash, or the default. Unknown hashes are ignored. */
function tabFromHash(): TabId {
  const h = window.location.hash.replace(/^#/, '')
  return (IDS.includes(h) ? h : 'core') as TabId
}

export default function App() {
  // The tab lives in the hash so a tab is linkable: "open the Worlds tab" is a
  // URL rather than an instruction, a reload keeps your place, and the browser
  // back button steps between tabs instead of leaving the app.
  const [tab, setTab] = useState<TabId>(tabFromHash)
  const mode = useAsync(() => api.authMode(), [])

  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  function open(id: TabId) {
    setTab(id)
    // Assigning the hash rather than pushState so the back button works, and
    // so the hashchange listener above stays the single place tab state is set.
    window.location.hash = id
  }

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
          Generate entities and tiles, build spritesheets from them, and teach the
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
              onClick={() => open(t.id)}
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
        {tab === 'ref-map' && <ReferenceTab kind="map" />}
        {tab === 'tiles' && <Tiles />}
        {tab === 'maps' && <Maps />}
        {tab === 'worlds' && <Worlds />}
        {tab === 'training' && <Training />}
        {tab === 'gallery' && <Gallery />}
        {tab === 'commands' && <Commands />}
        {tab === 'settings' && <Settings onModeChange={mode.reload} />}
      </div>
    </>
  )
}
