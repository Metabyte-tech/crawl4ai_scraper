import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [url, setUrl] = useState('')
  const [query, setQuery] = useState('')
  const [isCrawling, setIsCrawling] = useState(false)
  const [isThinking, setIsThinking] = useState(false)
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState('')

  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleCrawl = async (isDeep = false) => {
    if (!url) return
    setIsCrawling(true)
    setStatus(isDeep ? 'Performing deep crawl (multi-page)... this may take a minute.' : 'Crawling and ingesting content...')
    try {
      const endpoint = isDeep ? 'http://localhost:8000/crawl/deep' : 'http://localhost:8000/crawl'
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      })
      const data = await resp.json()
      setStatus(data.message || 'Crawl successful!')
      if (data.status === 'success') {
        setUrl('')
      }
    } catch (err) {
      setStatus('Crawl failed: ' + err.message)
    } finally {
      setIsCrawling(false)
    }
  }

  const handleClear = async () => {
    if (!confirm('Are you sure you want to clear the AI memory? This will delete all previously ingested data.')) return

    setStatus('Clearing memory...')
    try {
      const resp = await fetch('http://localhost:8000/clear', {
        method: 'POST'
      })
      const data = await resp.json()
      setStatus(data.message || 'Memory cleared!')
      setMessages([])
    } catch (err) {
      setStatus('Clear failed: ' + err.message)
    }
  }

  const handleSend = async () => {
    if (!query) return
    const userMsg = { role: 'user', text: query }
    setMessages(prev => [...prev, userMsg])
    const currentQuery = query
    setQuery('')
    setIsThinking(true)

    try {
      const resp = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: currentQuery })
      })
      const data = await resp.json()
      const botMsg = { role: 'bot', text: data.response }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Error: ' + err.message }])
    } finally {
      setIsThinking(false)
    }
  }

  return (
    <div className="glass-card">
      <h1 className="vibrant-gradient-text">Crawl4AI Assistant</h1>

      <div className="input-group">
        <input
          type="text"
          placeholder="Enter website URL to ingest..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={isCrawling}
        />
        <div className="button-group">
          <button onClick={() => handleCrawl(false)} disabled={isCrawling || !url}>
            {isCrawling ? 'Processing...' : 'Ingest Site'}
          </button>
          <button className="secondary-btn" onClick={() => handleCrawl(true)} disabled={isCrawling || !url}>
            Deep Ingest
          </button>
          <button className="danger-btn" onClick={handleClear} disabled={isCrawling}>
            Clear Memory
          </button>
        </div>
      </div>

      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="bot-message message">
              Hello! I'm your AI assistant. Ingest a website above and ask me anything about it.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}-message`}>
              {m.text}
            </div>
          ))}
          {isThinking && (
            <div className="bot-message message">
              Thinking...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-group" style={{ marginBottom: 0 }}>
          <input
            type="text"
            placeholder="Ask a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            disabled={isThinking}
          />
          <button onClick={handleSend} disabled={isThinking || !query}>
            Send
          </button>
        </div>
      </div>

      {status && <div className="status-indicator">{status}</div>}
    </div>
  )
}

export default App
