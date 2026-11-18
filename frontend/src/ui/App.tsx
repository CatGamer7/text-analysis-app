import { useEffect, useRef, useState } from 'react'

const MODELS = ['Модель 1', 'Модель 2', 'Модель 3', 'Модель 4', 'Модель 5', 'Модель 6']

const RESULT_CARDS = ['Основные темы', 'Тональность', 'Ключевые цитаты', 'Связи между темами', 'Частотность упоминаний']

// В макете можно загрузить только такие файлы, остальные показывают ошибку
const SUPPORTED_FILE = /\.(docx|txt|csv|xlsx)$/i

type Mode = 'text' | 'file'
type UploadedFile = { name: string; failed: boolean }

// Запись истории: заголовок, время и всё, что нужно, чтобы снова открыть запрос
interface Analysis {
  id: number
  title: string
  time: string
  model: string
  mode: Mode
  text: string
  file: UploadedFile | null
  progress: number
}

function App() {
  const [model, setModel] = useState('')
  const [mode, setMode] = useState<Mode>('text')
  const [text, setText] = useState('')
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [progress, setProgress] = useState(0)
  const [showResults, setShowResults] = useState(false)
  // История живёт только пока открыто приложение
  const [history, setHistory] = useState<Analysis[]>([])
  const [currentId, setCurrentId] = useState<number | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const isLoading = file !== null && !file.failed && progress < 100
  const isLoaded = file !== null && !file.failed && progress === 100
  const canRun = mode === 'text' ? text.trim() !== '' : isLoaded

  // Имитация загрузки файла
  useEffect(() => {
    if (!isLoading) return
    const timer = setInterval(() => setProgress((p) => Math.min(p + 4, 100)), 100)
    return () => clearInterval(timer)
  }, [isLoading])

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const chosen = e.target.files?.[0]
    e.target.value = ''
    if (!chosen) return
    setFile({ name: chosen.name, failed: !SUPPORTED_FILE.test(chosen.name) })
    setProgress(0)
    setShowResults(false)
  }

  function handleRun() {
    const id = currentId ?? Date.now()
    const entry: Analysis = {
      id,
      title: mode === 'text' ? text.trim() : (file?.name ?? ''),
      time: new Date().toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' }),
      model,
      mode,
      text,
      file,
      progress,
    }
    // Повторный запуск обновляет открытую запись, новый запрос встаёт в начало истории
    setHistory((items) =>
      currentId === null ? [entry, ...items] : items.map((item) => (item.id === id ? entry : item)),
    )
    setCurrentId(id)
    setShowResults(true)
  }

  function handleNewAnalysis() {
    setCurrentId(null)
    setModel('')
    setMode('text')
    setText('')
    setFile(null)
    setProgress(0)
    setShowResults(false)
  }

  function handleOpenAnalysis(item: Analysis) {
    setCurrentId(item.id)
    setModel(item.model)
    setMode(item.mode)
    setText(item.text)
    setFile(item.file)
    setProgress(item.progress)
    setShowResults(true)
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">Я&amp;AI</div>

        <button className="new-analysis" onClick={handleNewAnalysis}>
          <span>+</span> Новый анализ
        </button>

        {history.length > 0 && <div className="history-title">История</div>}
        {history.map((item) => (
          <button
            key={item.id}
            className={`history-item${item.id === currentId ? ' active' : ''}`}
            onClick={() => handleOpenAnalysis(item)}
          >
            <span className="history-name">{item.title}</span>
            <span className="history-time">{item.time}</span>
          </button>
        ))}
      </aside>

      <main className="content">
        <section className="model-picker">
          <h2>Выберите тип анализа</h2>
          <div className="model-grid">
            {MODELS.map((name) => (
              <button
                key={name}
                type="button"
                className={`model-card${model === name ? ' selected' : ''}`}
                onClick={() => setModel(name)}
              >
                <span className="model-radio" aria-hidden="true" />
                {name}
              </button>
            ))}
          </div>
        </section>

        {showResults && (
          <>
            <h2>Статистика запроса</h2>
            <section className="results">
              {RESULT_CARDS.map((title) => (
                <article key={title} className={title === 'Ключевые цитаты' ? 'tall' : ''}>
                  <h3>{title}</h3>
                </article>
              ))}
            </section>
          </>
        )}

        <section className="composer">
          <div className="composer-top">
            <div className="mode-switch">
              <button className={mode === 'text' ? 'active' : ''} onClick={() => setMode('text')}>
                Текст
              </button>
              <button className={mode === 'file' ? 'active' : ''} onClick={() => setMode('file')}>
                Файл
              </button>
            </div>

            <button
              className="send"
              aria-label="Запустить анализ"
              disabled={!canRun}
              onClick={handleRun}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                <path fill="currentColor" d="M3 11.5 21 3l-8.5 18-2.2-7.3L3 11.5Z" />
              </svg>
            </button>
          </div>

          <input ref={fileInput} type="file" hidden onChange={handleFileChange} />

          {mode === 'text' && (
            <textarea
              rows={2}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Введите текст"
            />
          )}

          {mode === 'file' && !file && (
            <button className="choose-file" onClick={() => fileInput.current?.click()}>
              <FileIcon />
              Выберите файл
            </button>
          )}

          {mode === 'file' && file && (
            <div className="file-row">
              <FileIcon />
              <div className="file-info">
                <div className="file-top">
                  <span className="file-name">{file.name}</span>
                  {isLoading && <span>{progress}%</span>}
                  {isLoaded && <span>Загружено</span>}
                </div>
                {file.failed ? (
                  <div className="file-error">
                    Ошибка загрузки{' '}
                    <button onClick={() => fileInput.current?.click()}>Повторить</button>
                  </div>
                ) : (
                  <div className="progress">
                    <div style={{ width: `${progress}%` }} />
                  </div>
                )}
              </div>
              <button
                className="icon-button"
                aria-label="Убрать файл"
                onClick={() => {
                  setFile(null)
                  setShowResults(false)
                }}
              >
                ✕
              </button>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function FileIcon() {
  return (
    <svg className="file-icon" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path
        fill="currentColor"
        d="M6 2h8l6 6v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm7 2v5h5l-5-5ZM8 13v2h8v-2H8Zm0 4v2h8v-2H8Z"
      />
    </svg>
  )
}

export default App
