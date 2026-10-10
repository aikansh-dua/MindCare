(() => {
  const OWNER_KEY = "mindcare_owner_id";
  const ACTIVE_KEY = "mindcare_active_conversation";
  const THEME_KEY = "mindcare_theme";

  const app = document.getElementById("app");
  const welcome = document.getElementById("welcome");
  const chatShell = document.getElementById("chatShell");
  const startChat = document.getElementById("startChat");
  const newChatBtn = document.getElementById("newChatBtn");
  const sheetNewChat = document.getElementById("sheetNewChat");
  const homeBtn = document.getElementById("homeBtn");
  const openHelplines = document.getElementById("openHelplines");
  const helpDialog = document.getElementById("helpDialog");
  const momentsToggle = document.getElementById("momentsToggle");
  const closeMoments = document.getElementById("closeMoments");
  const momentsSheet = document.getElementById("momentsSheet");
  const sheetBackdrop = document.getElementById("sheetBackdrop");
  const momentsRail = document.getElementById("momentsRail");
  const recentsList = document.getElementById("recentsList");
  const momentsList = document.getElementById("momentsList");
  const recentsEmpty = document.getElementById("recentsEmpty");
  const chatTitle = document.getElementById("chatTitle");
  const thread = document.getElementById("thread");
  const composer = document.getElementById("composer");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const metaLine = document.getElementById("metaLine");
  const themeToggle = document.getElementById("themeToggle");
  const themeToggleLabel = document.getElementById("themeToggleLabel");
  const themeColorMeta = document.getElementById("themeColor");

  let busy = false;
  let conversations = [];
  let activeId = localStorage.getItem(ACTIVE_KEY) || null;

  function currentTheme() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function applyTheme(theme) {
    const next = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(THEME_KEY, next);
    if (themeToggleLabel) {
      themeToggleLabel.textContent = next === "dark" ? "Light" : "Velvet";
    }
    if (themeToggle) {
      themeToggle.setAttribute(
        "aria-label",
        next === "dark" ? "Switch to light eucalyptus theme" : "Switch to velvet dark theme"
      );
      themeToggle.title = next === "dark" ? "Light room" : "Velvet room";
    }
    if (themeColorMeta) {
      themeColorMeta.setAttribute("content", next === "dark" ? "#4F565C" : "#d9e8df");
    }
  }

  function toggleTheme() {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
  }

  applyTheme(localStorage.getItem(THEME_KEY) || currentTheme());

  function ownerId() {
    let id = localStorage.getItem(OWNER_KEY);
    if (!id) {
      id = `web_${crypto.randomUUID().slice(0, 10)}`;
      localStorage.setItem(OWNER_KEY, id);
    }
    return id;
  }

  function setActiveId(id) {
    activeId = id;
    if (id) localStorage.setItem(ACTIVE_KEY, id);
    else localStorage.removeItem(ACTIVE_KEY);
  }

  /** Flash-polish title when leaving a conversation (non-blocking). */
  function finalizeTitle(conversationId) {
    if (!conversationId) return;
    const url =
      `/conversations/${encodeURIComponent(conversationId)}/title` +
      `?owner_id=${encodeURIComponent(ownerId())}`;
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ owner_id: ownerId() }),
      keepalive: true,
    }).catch(() => {});
  }

  function leaveActiveConversation() {
    if (activeId) finalizeTitle(activeId);
  }

  function setMood(mood) {
    app.dataset.mood = mood || "idle";
  }

  function showHelplines() {
    if (typeof helpDialog.showModal === "function") helpDialog.showModal();
    else helpDialog.setAttribute("open", "");
  }

  function openMomentsSheet() {
    momentsSheet.hidden = false;
    sheetBackdrop.hidden = false;
  }

  function closeMomentsSheet() {
    momentsSheet.hidden = true;
    sheetBackdrop.hidden = true;
  }

  function showChatView() {
    welcome.hidden = true;
    chatShell.hidden = false;
    welcome.setAttribute("aria-hidden", "true");
    chatShell.setAttribute("aria-hidden", "false");
    app.dataset.view = "chat";
    setMood("idle");
    input.focus();
  }

  function showWelcomeView() {
    leaveActiveConversation();
    chatShell.hidden = true;
    welcome.hidden = false;
    chatShell.setAttribute("aria-hidden", "true");
    welcome.setAttribute("aria-hidden", "false");
    app.dataset.view = "welcome";
    chatTitle.textContent = "This moment";
    setMood("idle");
    closeMomentsSheet();
  }

  function renderEmpty() {
    thread.innerHTML = "";
    const hint = document.createElement("p");
    hint.className = "empty-hint";
    hint.textContent = "The room is quiet. Share whatever feels true.";
    thread.appendChild(hint);
  }

  function clearEmpty() {
    thread.querySelector(".empty-hint")?.remove();
  }

  function isCrisisText(text) {
    return /concerned about your safety|1860|9152987821/i.test(text || "");
  }

  function addMessage(role, text, { crisis = false } = {}) {
    clearEmpty();
    const wrap = document.createElement("article");
    wrap.className = `note note-${role}${crisis ? " crisis" : ""}`;

    const who = document.createElement("p");
    who.className = "note-who";
    who.textContent = role === "user" ? "You" : "MindCare";

    const body = document.createElement("p");
    body.className = "note-body";
    body.textContent = text;

    wrap.append(who, body);
    thread.appendChild(wrap);
    thread.scrollTop = thread.scrollHeight;
  }

  function renderThread(memory) {
    thread.innerHTML = "";
    if (!memory?.length) {
      renderEmpty();
      return;
    }
    memory.forEach((m, i) => {
      const role = m.role === "user" ? "user" : "bot";
      const wrap = document.createElement("article");
      wrap.className = `note note-${role}${
        role === "bot" && isCrisisText(m.text) ? " crisis" : ""
      }`;
      wrap.style.animationDelay = `${Math.min(i * 0.04, 0.35)}s`;
      const who = document.createElement("p");
      who.className = "note-who";
      who.textContent = role === "user" ? "You" : "MindCare";
      const body = document.createElement("p");
      body.className = "note-body";
      body.textContent = m.text || "";
      wrap.append(who, body);
      thread.appendChild(wrap);
    });
    thread.scrollTop = thread.scrollHeight;
  }

  function meaningfulConversations() {
    return conversations.filter(
      (c) => (c.message_count || 0) > 0 || (c.title && c.title !== "New conversation")
    );
  }

  function renderMoments() {
    const items = meaningfulConversations();
    recentsList.innerHTML = "";
    momentsList.innerHTML = "";

    momentsRail.hidden = items.length === 0;
    recentsEmpty.hidden = items.length > 0;

    items.forEach((c, index) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `moment-chip${c.id === activeId ? " active" : ""}`;
      chip.style.animationDelay = `${index * 0.05}s`;
      chip.textContent = c.title || "Untitled moment";
      chip.addEventListener("click", () => openConversation(c.id));
      recentsList.appendChild(chip);

      const row = document.createElement("div");
      row.className = `moment-row${c.id === activeId ? " active" : ""}`;
      const open = document.createElement("button");
      open.type = "button";
      open.className = "moment-open";
      open.textContent = c.title || "Untitled moment";
      open.addEventListener("click", () => {
        closeMomentsSheet();
        openConversation(c.id);
      });
      const del = document.createElement("button");
      del.type = "button";
      del.className = "moment-del";
      del.setAttribute("aria-label", "Remove moment");
      del.textContent = "×";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteConversation(c.id);
      });
      row.append(open, del);
      momentsList.appendChild(row);
    });
  }

  async function refreshConversations() {
    const res = await fetch(`/conversations?owner_id=${encodeURIComponent(ownerId())}`);
    if (!res.ok) return;
    const data = await res.json();
    conversations = data.conversations || [];
    renderMoments();
  }

  async function createConversation() {
    const res = await fetch("/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ owner_id: ownerId() }),
    });
    const session = await res.json();
    setActiveId(session.id);
    await refreshConversations();
    return session;
  }

  async function openConversation(id) {
    if (activeId && activeId !== id) leaveActiveConversation();
    setActiveId(id);
    showChatView();
    const meta = conversations.find((c) => c.id === id);
    chatTitle.textContent = meta?.title || "This moment";
    renderMoments();

    try {
      const res = await fetch(
        `/conversations/${encodeURIComponent(id)}?owner_id=${encodeURIComponent(ownerId())}`
      );
      if (!res.ok) {
        renderEmpty();
        return;
      }
      const data = await res.json();
      chatTitle.textContent = data.title || "This moment";
      renderThread(data.memory || []);
    } catch {
      renderEmpty();
    }
  }

  async function startNewMoment() {
    closeMomentsSheet();
    leaveActiveConversation();
    await createConversation();
    showChatView();
    chatTitle.textContent = "This moment";
    renderEmpty();
    renderMoments();
    input.focus();
  }

  async function deleteConversation(id) {
    await fetch(`/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ owner_id: ownerId() }),
    });
    if (activeId === id) {
      setActiveId(null);
      showWelcomeView();
      renderEmpty();
    }
    await refreshConversations();
  }

  function addTyping() {
    clearEmpty();
    const wrap = document.createElement("article");
    wrap.className = "note note-bot typing";
    wrap.id = "typingIndicator";
    wrap.innerHTML = `
      <p class="note-who">MindCare</p>
      <p class="note-body" aria-label="Listening">
        <span class="pulse-dot"></span>
        <span class="pulse-dot"></span>
        <span class="pulse-dot"></span>
      </p>`;
    thread.appendChild(wrap);
    thread.scrollTop = thread.scrollHeight;
  }

  function removeTyping() {
    document.getElementById("typingIndicator")?.remove();
  }

  function setBusy(state) {
    busy = state;
    sendBtn.disabled = state;
    input.disabled = state;
    setMood(state ? "listening" : "idle");
  }

  function setMeta(result) {
    if (!result) {
      metaLine.hidden = true;
      metaLine.textContent = "";
      return;
    }
    const parts = [];
    if (result.emotion) parts.push(result.emotion);
    if (result.crisis) {
      parts.push("care resources shared");
      setMood("crisis");
    }
    metaLine.textContent = parts.join(" · ");
    metaLine.hidden = parts.length === 0;
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  }

  async function sendMessage(text) {
    const message = text.trim();
    if (!message || busy) return;

    if (!activeId) await createConversation();
    showChatView();

    addMessage("user", message);
    input.value = "";
    autosize();
    setBusy(true);
    addTyping();
    setMeta(null);

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          owner_id: ownerId(),
          conversation_id: activeId,
          message,
        }),
      });
      const data = await res.json().catch(() => ({}));
      removeTyping();

      if (!res.ok) {
        addMessage("bot", data.error || "Something went quiet on my side. Try again in a moment.");
        return;
      }

      if (data.conversation_id) setActiveId(data.conversation_id);
      if (data.conversation_title) chatTitle.textContent = data.conversation_title;

      setMood("speaking");
      addMessage("bot", data.response || "…", { crisis: Boolean(data.crisis) });
      setMeta(data);
      await refreshConversations();
      setTimeout(() => {
        if (!busy) setMood(data.crisis ? "crisis" : "idle");
      }, 1200);
    } catch {
      removeTyping();
      addMessage("bot", "I couldn’t reach the room’s server. Is MindCare running?");
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  startChat.addEventListener("click", startNewMoment);
  newChatBtn.addEventListener("click", startNewMoment);
  sheetNewChat.addEventListener("click", startNewMoment);
  homeBtn.addEventListener("click", showWelcomeView);
  openHelplines.addEventListener("click", showHelplines);
  momentsToggle.addEventListener("click", openMomentsSheet);
  closeMoments.addEventListener("click", closeMomentsSheet);
  sheetBackdrop.addEventListener("click", closeMomentsSheet);
  themeToggle?.addEventListener("click", toggleTheme);

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  refreshConversations();
  window.addEventListener("pagehide", leaveActiveConversation);
})();
