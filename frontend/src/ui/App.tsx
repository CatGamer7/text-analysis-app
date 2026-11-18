import { useEffect, useRef, useState } from 'react'

const API_BASE = 'http://127.0.0.1:8000/api/v1'

interface ModelInfo {
  id: string
  name: string
  description: string
  task_type: string
  languages: string[]
  labels: string[]
}

const DEFAULT_MODELS: ModelInfo[] = [
  {
    id: 'rubert-sentiment-conversational',
    name: 'RuBERT Conversational Sentiment',
    description: 'Анализ тональности текстов на базе RuBERT conversational (ONNX).',
    task_type: 'sentiment',
    languages: ['ru'],
    labels: ['negative', 'neutral', 'positive'],
  },
  {
    id: 'rubert-tiny2-sentiment',
    name: 'RuBERT Tiny2 Sentiment (230MB)',
    description: 'Полная модель RuBERT-tiny2 (~230 МБ) для глубокого анализа соцсетей.',
    task_type: 'sentiment',
    languages: ['ru'],
    labels: ['neutral', 'positive', 'negative'],
  },
  {
    id: 'roberta-sentiment-en',
    name: 'Twitter-RoBERTa Sentiment (English)',
    description: 'Анализ тональности англоязычных твитов и постов в соцсетях.',
    task_type: 'sentiment',
    languages: ['en'],
    labels: ['negative', 'neutral', 'positive'],
  },
  {
    id: 'rubert-toxicity-ru',
    name: 'RuBERT Toxicity & Hate Speech',
    description: 'Детекция токсичности, оскорблений и агрессии в комментариях.',
    task_type: 'toxicity',
    languages: ['ru'],
    labels: ['neutral', 'toxic', 'insult', 'threat', 'obscenity'],
  },
  {
    id: 'rubert-emotion-ru',
    name: 'RuBERT Emotion Classifier',
    description: 'Классификация спектра базовых эмоций в пользовательском контенте.',
    task_type: 'emotion',
    languages: ['ru'],
    labels: ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise'],
  },
  {
    id: 'distilbert-emotion-en',
    name: 'DistilBERT Emotion (English)',
    description: 'Экспресс-классификация эмоций на английском языке.',
    task_type: 'emotion',
    languages: ['en'],
    labels: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
  },
]

const SUPPORTED_FILE = /\.(docx|txt|csv|xlsx)$/i

type Mode = 'text' | 'file'
type UploadedFile = { name: string; failed: boolean; raw?: File }

interface LabelScore {
  label: string
  score: number
  percentage: number
}

interface ProbeResult {
  model_id: string
  text: string
  primary_label: string
  confidence: number
  scores: LabelScore[]
  latency_ms: number
  tokens_count: number
  device: string
  is_mock?: boolean
}

interface TaskResult {
  task_id: string
  filename: string
  total_rows: number
  processed_rows: number
  download_url: string
}

interface AnalysisEntry {
  id: number
  title: string
  time: string
  modelId: string
  mode: Mode
  text: string
  file: UploadedFile | null
  progress: number
  probeResult?: ProbeResult | null
  taskResult?: TaskResult | null
}

function App() {
  const [models, setModels] = useState<ModelInfo[]>(DEFAULT_MODELS)
  const [selectedModelId, setSelectedModelId] = useState<string>(DEFAULT_MODELS[0].id)
  const [mode, setMode] = useState<Mode>('text')
  const [text, setText] = useState('')
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [progress, setProgress] = useState(0)
  const [showResults, setShowResults] = useState(false)
  const [serverOnline, setServerOnline] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  const [currentProbe, setCurrentProbe] = useState<ProbeResult | null>(null)
  const [currentTask, setCurrentTask] = useState<TaskResult | null>(null)

  const [history, setHistory] = useState<AnalysisEntry[]>([])
  const [currentId, setCurrentId] = useState<number | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  // Проверка статуса бэкенда и загрузка каталога моделей
  useEffect(() => {
    let mounted = true

    async function checkHealthAndModels() {
      try {
        const resHealth = await fetch(`${API_BASE}/health`)
        if (resHealth.ok) {
          if (mounted) setServerOnline(true)
          const resModels = await fetch(`${API_BASE}/models`)
          if (resModels.ok && mounted) {
            const json = await resModels.json()
            const data: ModelInfo[] = Array.isArray(json) ? json : json.items || []
            if (data.length > 0) {
              setModels(data)
              if (!selectedModelId) {
                setSelectedModelId(data[0].id)
              }
            }
          }
        } else {
          if (mounted) setServerOnline(false)
        }
      } catch {
        if (mounted) setServerOnline(false)
      }
    }

    checkHealthAndModels()
    const timer = setInterval(checkHealthAndModels, 3000)
    return () => {
      mounted = false
      clearInterval(timer)
    }
  }, [selectedModelId])

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const chosen = e.target.files?.[0]
    e.target.value = ''
    if (!chosen) return
    const isSupported = SUPPORTED_FILE.test(chosen.name)
    setFile({ name: chosen.name, failed: !isSupported, raw: chosen })
    setProgress(0)
    setShowResults(false)
    setCurrentTask(null)
  }

  async function handleRun() {
    setIsProcessing(true)
    const id = currentId ?? Date.now()

    if (mode === 'text') {
      try {
        const payload = { text: text.trim() }
        const res = await fetch(`${API_BASE}/models/${selectedModelId}/probe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })

        if (!res.ok) {
          throw new Error(`Ошибка инференса: ${res.statusText}`)
        }

        const probeData: ProbeResult = await res.json()
        setCurrentProbe(probeData)
        setCurrentTask(null)
        setShowResults(true)

        const entry: AnalysisEntry = {
          id,
          title: text.trim().slice(0, 32) + (text.length > 32 ? '...' : ''),
          time: new Date().toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' }),
          modelId: selectedModelId,
          mode: 'text',
          text,
          file: null,
          progress: 100,
          probeResult: probeData,
        }

        setHistory((items) =>
          currentId === null ? [entry, ...items] : items.map((it) => (it.id === id ? entry : it)),
        )
        setCurrentId(id)
      } catch (err) {
        alert(String(err))
      } finally {
        setIsProcessing(false)
      }
    } else if (mode === 'file' && file?.raw) {
      try {
        const formData = new FormData()
        formData.append('file', file.raw)
        formData.append('model_id', selectedModelId)

        setProgress(5)
        const uploadRes = await fetch(`${API_BASE}/files/process`, {
          method: 'POST',
          body: formData,
        })

        if (!uploadRes.ok) {
          throw new Error(`Ошибка загрузки файла: ${uploadRes.statusText}`)
        }

        const uploadData = await uploadRes.json()
        const taskId = uploadData.task_id

        // Опрос статуса прогресса
        let isDone = false
        while (!isDone) {
          await new Promise((r) => setTimeout(r, 400))
          const progRes = await fetch(`${API_BASE}/files/tasks/${taskId}/progress`)
          if (progRes.ok) {
            const progData = await progRes.json()
            const pct = Math.max(5, Math.min(progData.progress_percent || 0, 100))
            setProgress(pct)

            if (progData.status === 'COMPLETED' || progData.status === 'SUCCESS') {
              isDone = true
              setProgress(100)
              const taskObj: TaskResult = {
                task_id: taskId,
                filename: file.name,
                total_rows: progData.total_rows || 0,
                processed_rows: progData.processed_rows || 0,
                download_url: `${API_BASE}/files/tasks/${taskId}/download`,
              }
              setCurrentTask(taskObj)
              setCurrentProbe(null)
              setShowResults(true)

              const entry: AnalysisEntry = {
                id,
                title: file.name,
                time: new Date().toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' }),
                modelId: selectedModelId,
                mode: 'file',
                text: '',
                file,
                progress: 100,
                taskResult: taskObj,
              }

              setHistory((items) =>
                currentId === null ? [entry, ...items] : items.map((it) => (it.id === id ? entry : it)),
              )
              setCurrentId(id)
            } else if (progData.status === 'FAILED') {
              throw new Error(progData.error || 'Ошибка при обработке файла.')
            }
          }
        }
      } catch (err) {
        alert(String(err))
      } finally {
        setIsProcessing(false)
      }
    }
  }

  function handleNewAnalysis() {
    setCurrentId(null)
    setText('')
    setFile(null)
    setProgress(0)
    setShowResults(false)
    setCurrentProbe(null)
    setCurrentTask(null)
  }

  function handleOpenAnalysis(item: AnalysisEntry) {
    setCurrentId(item.id)
    setSelectedModelId(item.modelId)
    setMode(item.mode)
    setText(item.text)
    setFile(item.file)
    setProgress(item.progress)
    setCurrentProbe(item.probeResult ?? null)
    setCurrentTask(item.taskResult ?? null)
    setShowResults(true)
  }

  const canRun =
    !isProcessing &&
    (mode === 'text' ? text.trim().length > 0 : file !== null && !file.failed)

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">Я&amp;AI</div>

        <div className="server-status">
          <span className={`status-dot ${serverOnline ? 'online' : ''}`} />
          <span>{serverOnline ? 'Сервер активен (8000)' : 'Подключение к серверу...'}</span>
        </div>

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
            {models.slice(0, 6).map((m) => {
              const isSelected = selectedModelId === m.id
              return (
                <button
                  key={m.id}
                  type="button"
                  className={`model-card${isSelected ? ' selected' : ''}`}
                  onClick={() => setSelectedModelId(m.id)}
                >
                  <span className="model-radio" aria-hidden="true" />
                  <div className="model-title-wrap">
                    <span className="model-title">{m.name}</span>
                    <span className="model-sub">{m.description}</span>
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        {showResults && (
          <>
            <h2>Статистика запроса</h2>
            <section className="results">
              {/* Карточка 1: Тональность */}
              <article>
                <h3>Тональность</h3>
                <div className="card-content">
                  {currentProbe ? (
                    <>
                      <div className="card-primary-class">{currentProbe.primary_label}</div>
                      <div className="card-confidence">
                        Уверенность модели: {(currentProbe.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="score-bars">
                        {currentProbe.scores.map((s) => (
                          <div key={s.label} className="score-bar-row">
                            <span className="score-bar-label">{s.label}</span>
                            <div className="score-bar-track">
                              <div
                                className="score-bar-fill"
                                style={{ width: `${s.percentage}%` }}
                              />
                            </div>
                            <span className="score-bar-pct">{s.percentage.toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : currentTask ? (
                    <>
                      <div className="card-primary-class">CSV ОБРАБОТАН</div>
                      <div className="card-confidence">
                        Все строки размечены выбранной моделью
                      </div>
                      <a
                        href={currentTask.download_url}
                        download
                        className="download-link"
                      >
                        Скачать готовый CSV
                      </a>
                    </>
                  ) : (
                    <div>Нет данных</div>
                  )}
                </div>
              </article>

              {/* Карточка 2: Основные темы */}
              <article>
                <h3>Основные темы</h3>
                <div className="card-content">
                  {currentProbe ? (
                    <div style={{ lineHeight: 1.6 }}>
                      Семантическая ориентация:{' '}
                      <strong style={{ color: 'var(--text)' }}>
                        {currentProbe.primary_label.toUpperCase()}
                      </strong>
                      <br />
                      Языковая среда: Социальные медиа (RU / EN)
                      <br />
                      Режим классификации: Прямой тензорный инференс
                    </div>
                  ) : currentTask ? (
                    <div>Массив текстов из файла: {currentTask.filename}</div>
                  ) : (
                    <div>—</div>
                  )}
                </div>
              </article>

              {/* Карточка 3 (tall): Ключевые цитаты */}
              <article className="tall">
                <h3>Ключевые цитаты</h3>
                <div className="card-content">
                  {currentProbe ? (
                    <div className="quote-box">«{currentProbe.text}»</div>
                  ) : currentTask ? (
                    <div style={{ lineHeight: 1.6 }}>
                      Файл: <strong>{currentTask.filename}</strong>
                      <br />
                      Всего строк в массиве: {currentTask.total_rows}
                      <br />
                      Успешно обработано: {currentTask.processed_rows}
                    </div>
                  ) : (
                    <div>—</div>
                  )}
                </div>
              </article>

              {/* Карточка 4: Связи между темами / Телеметрия */}
              <article>
                <h3>Связи между темами</h3>
                <div className="card-content">
                  {currentProbe ? (
                    <>
                      <div className="telemetry-row">
                        <span className="telemetry-label">Время инференса:</span>
                        <span className="telemetry-val">{currentProbe.latency_ms} мс</span>
                      </div>
                      <div className="telemetry-row">
                        <span className="telemetry-label">Вычислитель:</span>
                        <span className="telemetry-val">{currentProbe.device}</span>
                      </div>
                      <div className="telemetry-row">
                        <span className="telemetry-label">Модель ID:</span>
                        <span className="telemetry-val" style={{ fontSize: 10 }}>
                          {currentProbe.model_id}
                        </span>
                      </div>
                    </>
                  ) : currentTask ? (
                    <div className="telemetry-row">
                      <span className="telemetry-label">ID задачи:</span>
                      <span className="telemetry-val" style={{ fontSize: 10 }}>
                        {currentTask.task_id.slice(0, 12)}...
                      </span>
                    </div>
                  ) : (
                    <div>—</div>
                  )}
                </div>
              </article>

              {/* Карточка 5: Частотность упоминаний */}
              <article>
                <h3>Частотность упоминаний</h3>
                <div className="card-content">
                  {currentProbe ? (
                    <>
                      <div className="telemetry-row">
                        <span className="telemetry-label">Количество токенов:</span>
                        <span className="telemetry-val">{currentProbe.tokens_count}</span>
                      </div>
                      <div className="telemetry-row">
                        <span className="telemetry-label">Длина символов:</span>
                        <span className="telemetry-val">{currentProbe.text.length}</span>
                      </div>
                    </>
                  ) : currentTask ? (
                    <div className="telemetry-row">
                      <span className="telemetry-label">Строк обработано:</span>
                      <span className="telemetry-val">100%</span>
                    </div>
                  ) : (
                    <div>—</div>
                  )}
                </div>
              </article>
            </section>
          </>
        )}

        <section className="composer">
          <div className="composer-top">
            <div className="mode-switch">
              <button
                className={mode === 'text' ? 'active' : ''}
                onClick={() => setMode('text')}
              >
                Текст
              </button>
              <button
                className={mode === 'file' ? 'active' : ''}
                onClick={() => setMode('file')}
              >
                Файл
              </button>
            </div>

            <button
              className="send"
              aria-label="Запустить анализ"
              disabled={!canRun}
              onClick={handleRun}
            >
              {isProcessing ? (
                <span style={{ fontSize: 12 }}>...</span>
              ) : (
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <path fill="currentColor" d="M3 11.5 21 3l-8.5 18-2.2-7.3L3 11.5Z" />
                </svg>
              )}
            </button>
          </div>

          <input ref={fileInput} type="file" hidden onChange={handleFileChange} />

          {mode === 'text' && (
            <textarea
              rows={2}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Введите текст для анализа..."
            />
          )}

          {mode === 'file' && !file && (
            <button className="choose-file" onClick={() => fileInput.current?.click()}>
              <FileIcon />
              Выберите CSV-файл для анализа
            </button>
          )}

          {mode === 'file' && file && (
            <div className="file-row">
              <FileIcon />
              <div className="file-info">
                <div className="file-top">
                  <span className="file-name">{file.name}</span>
                  {progress < 100 && <span>{progress}%</span>}
                  {progress === 100 && <span>Готово</span>}
                </div>
                {file.failed ? (
                  <div className="file-error">
                    Ошибка формата (поддерживаются .csv, .txt, .docx, .xlsx){' '}
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
                  setProgress(0)
                  setCurrentTask(null)
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
