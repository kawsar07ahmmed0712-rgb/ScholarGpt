const $ = (selector, root = document) =>
    root.querySelector(selector);

const $$ = (selector, root = document) =>
    [...root.querySelectorAll(selector)];


const DEFAULT_PREFS = {
    name:
        "Kawsar Ahmmed",

    role:
        "AI / ML Learner",

    model:
        "openai/gpt-oss-120b",

    answerStyle:
        "detailed",

    reasoning:
        "auto"
};


const state = {
    conversationId:
        null,

    conversations:
        [],

    models:
        [],

    busy:
        false,

    sourceSets:
        new Map(),

    activeSourceSet:
        null,

    activeSourceIndex:
        0,

    prefs: {
        ...DEFAULT_PREFS,

        ...safeJson(
            localStorage.getItem(
                "ragscholar_settings"
            ),
            {}
        )
    }
};


const els = {
    appShell:
        $("#appShell"),

    messages:
        $("#messages"),

    welcome:
        $("#welcome"),

    chatScroll:
        $("#chatScroll"),

    form:
        $("#composer"),

    input:
        $("#messageInput"),

    send:
        $("#sendBtn"),

    newChat:
        $("#newChatBtn"),

    search:
        $("#chatSearch"),

    history:
        $("#historyList"),

    modelCard:
        $("#modelCard"),

    modelName:
        $("#modelName"),

    modelMenu:
        $("#modelMenu"),

    modelList:
        $("#modelList"),

    settingsCard:
        $("#settingsCard"),

    settingsModal:
        $("#settingsModal"),

    settingsClose:
        $("#settingsClose"),

    settingsCancel:
        $("#settingsCancel"),

    settingsSave:
        $("#settingsSave"),

    profileName:
        $("#profileName"),

    profileRole:
        $("#profileRole"),

    answerStyle:
        $("#answerStyle"),

    reasoningMode:
        $("#reasoningMode"),

    settingsModel:
        $("#settingsModel"),

    userAvatar:
        $("#userAvatar"),

    userName:
        $("#userName"),

    drawer:
        $("#sourceDrawer"),

    drawerBody:
        $("#sourceBody"),

    drawerClose:
        $("#sourceClose"),

    drawerPrev:
        $("#sourcePrev"),

    drawerNext:
        $("#sourceNext"),

    drawerCount:
        $("#sourceCount"),

    corpusCount:
        $("#corpusCount"),

    runtimeStatus:
        $("#runtimeStatus")
};


function safeJson(
    value,
    fallback
) {
    try {
        return (
            JSON.parse(value)
            ?? fallback
        );

    } catch {
        return fallback;
    }
}


function escapeHtml(value) {
    return String(
        value ?? ""
    ).replace(
        /[&<>"']/g,

        ch => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        })[ch]
    );
}


async function api(
    url,
    options = {}
) {
    const headers = {
        ...(options.headers || {})
    };

    if (
        options.body
        && !headers["Content-Type"]
    ) {
        headers["Content-Type"] =
            "application/json";
    }

    const response = await fetch(
        url,
        {
            ...options,
            headers
        }
    );

    if (!response.ok) {
        let message =
            `HTTP ${response.status}`;

        try {
            const body =
                await response.json();

            message =
                body.error
                || message;

        } catch {}

        throw new Error(
            message
        );
    }

    return response;
}


function initials(name) {
    return (
        name
        || "U"
    )
        .trim()
        .split(/\s+/)
        .slice(0, 2)
        .map(
            x => x[0]
        )
        .join("")
        .toUpperCase();
}


function friendlyModel(id) {
    const model =
        state.models.find(
            x => x.id === id
        );

    if (model?.label) {
        return model.label;
    }

    const lower =
        String(
            id || ""
        ).toLowerCase();

    if (
        lower.includes(
            "gpt-oss-120b"
        )
    ) {
        return "GPT-OSS 120B";
    }

    if (
        lower.includes(
            "qwen"
        )
    ) {
        return "Qwen";
    }

    if (
        lower.includes(
            "deepseek"
        )
    ) {
        return "DeepSeek";
    }

    if (
        lower.includes(
            "llama"
        )
    ) {
        return "Llama";
    }

    return String(
        id || "Model"
    )
        .split("/")
        .pop();
}


function savePrefs() {
    localStorage.setItem(
        "ragscholar_settings",
        JSON.stringify(
            state.prefs
        )
    );

    syncPrefsUI();
}


function syncPrefsUI() {
    els.userAvatar.textContent =
        initials(
            state.prefs.name
        );

    els.userName.textContent =
        state.prefs.name
        || "User";

    els.modelName.textContent =
        friendlyModel(
            state.prefs.model
        );

    els.profileName.value =
        state.prefs.name
        || "";

    els.profileRole.value =
        state.prefs.role
        || "";

    els.answerStyle.value =
        state.prefs.answerStyle
        || "balanced";

    els.reasoningMode.value =
        state.prefs.reasoning
        || "auto";

    if (
        [
            ...els.settingsModel.options
        ].some(
            option =>
                option.value
                === state.prefs.model
        )
    ) {
        els.settingsModel.value =
            state.prefs.model;
    }
}


function normalizeModel(model) {
    if (
        typeof model
        === "string"
    ) {
        return {
            id:
                model,

            label:
                friendlyModel(
                    model
                ),

            description:
                "",

            available:
                true
        };
    }

    const id =
        model.id
        || model.name
        || "";

    return {
        id,

        label:
            model.label
            || model.display_name
            || model.name
            || friendlyModel(id),

        description:
            model.description
            || "",

        available:
            model.available
            !== false
    };
}


async function loadModels() {
    try {
        const response =
            await api(
                "/api/models"
            );

        const body =
            await response.json();

        state.models =
            (
                body.models
                || []
            )
            .map(
                normalizeModel
            )
            .filter(
                x => x.id
            );

    } catch {
        state.models = [
            {
                id:
                    state.prefs.model,

                label:
                    friendlyModel(
                        state.prefs.model
                    ),

                available:
                    true
            }
        ];
    }

    if (
        !state.models.some(
            x =>
                x.id
                === state.prefs.model
        )
    ) {
        state.prefs.model =
            state.models.find(
                x => x.available
            )?.id

            || state.models[0]?.id

            || DEFAULT_PREFS.model;
    }

    renderModels();

    syncPrefsUI();
}


function renderModels() {
    els.modelList.innerHTML =
        state.models.map(
            model => `

            <button
                class="
                    model-option
                    ${
                        model.id
                        === state.prefs.model
                        ? "selected"
                        : ""
                    }
                "
                data-model="${escapeHtml(model.id)}"
                ${
                    model.available
                    ? ""
                    : "disabled"
                }
            >

                <span class="model-radio"></span>

                <span class="model-option-copy">

                    <strong>
                        ${escapeHtml(model.label)}
                    </strong>

                    <small>
                        ${
                            escapeHtml(
                                model.description
                                || "Groq model"
                            )
                        }
                    </small>

                </span>

                ${
                    model.id
                    === state.prefs.model

                    ? '<i data-lucide="check"></i>'

                    : ""
                }

            </button>

        `
        ).join("");

    els.settingsModel.innerHTML =
        state.models
        .filter(
            x => x.available
        )
        .map(
            model => `

            <option
                value="${escapeHtml(model.id)}"
            >
                ${escapeHtml(model.label)}
            </option>

        `
        )
        .join("");

    if (
        [
            ...els.settingsModel.options
        ].some(
            option =>
                option.value
                === state.prefs.model
        )
    ) {
        els.settingsModel.value =
            state.prefs.model;
    }

    $$(".model-option", els.modelList)
        .forEach(
            button => {
                button.addEventListener(
                    "click",
                    () => {
                        state.prefs.model =
                            button.dataset.model;

                        savePrefs();

                        renderModels();

                        hideModelMenu();
                    }
                );
            }
        );

    lucide.createIcons();
}


async function loadHealth() {
    try {
        const response =
            await api(
                "/api/health"
            );

        const body =
            await response.json();

        els.corpusCount.textContent =
            `${
                body.documents
                ?? 0
            } documents indexed`;

        if (
            body.runtime_ready
        ) {
            els.runtimeStatus.textContent =
                `${
                    body.device
                    ?.toUpperCase()
                    || "CPU"
                } · Ready`;

            els.runtimeStatus.classList.remove(
                "error"
            );

        } else {
            els.runtimeStatus.textContent =
                "Runtime error";

            els.runtimeStatus.classList.add(
                "error"
            );

            if (
                body.runtime_error
            ) {
                console.error(
                    body.runtime_error
                );
            }
        }

    } catch {
        els.runtimeStatus.textContent =
            "Offline";

        els.runtimeStatus.classList.add(
            "error"
        );
    }
}


async function loadConversations() {
    try {
        const response =
            await api(
                "/api/conversations"
            );

        const body =
            await response.json();

        state.conversations =
            body.conversations
            || [];

    } catch (error) {
        console.error(
            "Could not load conversations:",
            error
        );

        state.conversations =
            [];
    }

    renderHistory();
}


function renderHistory() {
    const needle =
        els.search.value
        .trim()
        .toLowerCase();

    const rows =
        state.conversations.filter(
            conversation =>
                !needle

                || String(
                    conversation.title
                    || ""
                )
                .toLowerCase()
                .includes(
                    needle
                )
        );

    if (!rows.length) {
        els.history.innerHTML = `
            <div class="history-empty">
                ${
                    needle
                    ? "No matching conversations"
                    : "No conversations yet"
                }
            </div>
        `;

        return;
    }

    els.history.innerHTML =
        rows.map(
            conversation => `

            <div
                class="
                    history-row-wrap
                    ${
                        conversation.id
                        === state.conversationId
                        ? "active"
                        : ""
                    }
                "
            >

                <button
                    class="history-row"
                    data-id="${escapeHtml(conversation.id)}"
                >

                    <i data-lucide="message-square"></i>

                    <span>
                        ${
                            escapeHtml(
                                conversation.title
                                || "New chat"
                            )
                        }
                    </span>

                </button>


                <button
                    class="history-delete"
                    data-delete-id="${escapeHtml(conversation.id)}"
                    title="Delete conversation"
                >

                    <i data-lucide="trash-2"></i>

                </button>

            </div>

        `
        ).join("");

    $$(".history-row", els.history)
        .forEach(
            button => {
                button.addEventListener(
                    "click",
                    () =>
                        openConversation(
                            button.dataset.id
                        )
                );
            }
        );

    $$(".history-delete", els.history)
        .forEach(
            button => {
                button.addEventListener(
                    "click",
                    async event => {
                        event.stopPropagation();

                        await deleteConversation(
                            button.dataset.deleteId
                        );
                    }
                );
            }
        );

    lucide.createIcons();
}


async function deleteConversation(id) {
    try {
        await api(
            `/api/conversations/${encodeURIComponent(id)}`,
            {
                method:
                    "DELETE"
            }
        );

        if (
            state.conversationId
            === id
        ) {
            newChat();
        }

        await loadConversations();

    } catch (error) {
        console.error(
            error
        );
    }
}


function newChat() {
    state.conversationId =
        null;

    state.sourceSets.clear();

    els.messages.innerHTML =
        "";

    els.welcome.classList.remove(
        "hidden"
    );

    closeSourceDrawer();

    renderHistory();

    els.input.focus();
}


async function openConversation(id) {
    try {
        const response =
            await api(
                `/api/conversations/${encodeURIComponent(id)}`
            );

        const body =
            await response.json();

        state.conversationId =
            id;

        state.sourceSets.clear();

        els.messages.innerHTML =
            "";

        els.welcome.classList.add(
            "hidden"
        );

        closeSourceDrawer();

        for (
            const message
            of body.messages
            || []
        ) {
            if (
                message.role
                === "user"
            ) {
                appendUserMessage(
                    message.content
                );

            } else if (
                message.role
                === "assistant"
            ) {
                appendAssistantMessage(
                    message.content,

                    message.sources
                    || [],

                    message.meta
                    || {}
                );
            }
        }

        renderHistory();

        scrollToBottom(
            false
        );

    } catch (error) {
        console.error(
            error
        );
    }
}


// ============================================================
// Markdown renderer
// ============================================================

marked.setOptions({
    gfm:
        true,

    breaks:
        false,

    headerIds:
        false,

    mangle:
        false
});


function renderMarkdown(
    markdown,
    messageKey
) {
    let text =
        String(
            markdown
            || ""
        );

    const codeBlocks =
        [];

    const mathBlocks =
        [];

    const citationBlocks =
        [];


    // Protect code first.
    text = text.replace(
        /```[\s\S]*?```/g,

        block => {
            const token =
                `RAGSCHOLARCODEx${codeBlocks.length}xTOKEN`;

            const html =
                DOMPurify.sanitize(
                    marked.parse(
                        block
                    )
                );

            codeBlocks.push(
                html
            );

            return token;
        }
    );


    const stashMath =
        raw => {
            const token =
                `RAGSCHOLARMATHx${mathBlocks.length}xTOKEN`;

            mathBlocks.push(
                escapeHtml(
                    raw
                )
            );

            return token;
        };


    text = text.replace(
        /\$\$[\s\S]*?\$\$/g,
        stashMath
    );

    text = text.replace(
        /\\\[[\s\S]*?\\\]/g,
        stashMath
    );

    text = text.replace(
        /\\\([\s\S]*?\\\)/g,
        stashMath
    );

    text = text.replace(
        /\$(?!\s)([^$\n]+?)(?<!\s)\$/g,
        stashMath
    );


    // Transform [1], [2] ... into source buttons.
    text = text.replace(
        /\[(\d+)\]/g,

        (_, number) => {
            const token =
                `RAGSCHOLARCITEx${citationBlocks.length}xTOKEN`;

            const index =
                Number(number)
                - 1;

            citationBlocks.push(
                `
                <button
                    class="inline-citation"
                    data-source-set="${escapeHtml(messageKey)}"
                    data-source-index="${index}"
                    title="Open source ${number}"
                >
                    ${number}
                </button>
                `
            );

            return token;
        }
    );


    let html =
        DOMPurify.sanitize(
            marked.parse(
                text
            ),

            {
                ADD_ATTR: [
                    "target",
                    "rel"
                ]
            }
        );


    codeBlocks.forEach(
        (value, index) => {
            html = html.replace(
                `RAGSCHOLARCODEx${index}xTOKEN`,
                value
            );
        }
    );


    mathBlocks.forEach(
        (value, index) => {
            html = html.replace(
                `RAGSCHOLARMATHx${index}xTOKEN`,
                value
            );
        }
    );


    citationBlocks.forEach(
        (value, index) => {
            html = html.replace(
                `RAGSCHOLARCITEx${index}xTOKEN`,
                value
            );
        }
    );


    return html;
}


function typeset(element) {
    if (
        window.MathJax
        ?.typesetPromise
    ) {
        window.MathJax
        .typesetPromise(
            [element]
        )
        .catch(
            error =>
                console.warn(
                    "MathJax:",
                    error
                )
        );
    }
}


// ============================================================
// Messages
// ============================================================

function appendUserMessage(text) {
    els.welcome.classList.add(
        "hidden"
    );

    const row =
        document.createElement(
            "div"
        );

    row.className =
        "message-row user-row";

    row.innerHTML = `

        <div class="user-message">
            ${
                escapeHtml(text)
                .replace(
                    /\n/g,
                    "<br>"
                )
            }
        </div>

    `;

    els.messages.appendChild(
        row
    );
}


function renderVisualResults(
    visualResults,
    sourceSetKey
) {
    const visuals = (
        visualResults
        || []
    )
    .filter(
        source =>
            source
            && source.asset_url
    )
    .slice(
        0,
        5
    );

    if (!visuals.length) {
        return "";
    }

    const cards =
        visuals.map(
            (source, index) => {
                const caption =
                    source.caption
                    || source.label
                    || "Retrieved visual";

                const title =
                    source.title
                    || "Academic source";

                return `

                <button
                    class="visual-result-card"
                    data-source-set="${escapeHtml(sourceSetKey)}"
                    data-source-index="${index}"
                    type="button"
                >

                    <div class="visual-result-image-wrap">

                        <img
                            class="visual-result-image"
                            src="${escapeHtml(source.asset_url)}"
                            alt="${escapeHtml(caption)}"
                            loading="lazy"
                        >

                    </div>


                    <div class="visual-result-info">

                        <strong>
                            ${escapeHtml(caption)}
                        </strong>


                        <span>
                            ${escapeHtml(title)}
                            ${
                                source.page

                                ? ` · p. ${escapeHtml(source.page)}`

                                : ""
                            }
                        </span>

                    </div>

                </button>

                `;
            }
        )
        .join("");


    return `

        <section class="visual-results">

            <div class="visual-results-heading">

                <div>

                    <i data-lucide="images"></i>

                    <span>
                        Visual results
                    </span>

                </div>


                <small>
                    ${visuals.length}
                    retrieved
                </small>

            </div>


            <div class="visual-results-track">

                ${cards}

            </div>

        </section>

    `;
}


function appendAssistantMessage(
    markdown,
    sources = [],
    meta = {}
) {
    els.welcome.classList.add(
        "hidden"
    );

    const key =
        `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    const visualKey =
        `${key}_visuals`;

    const visualResults =
        Array.isArray(
            meta.visual_results
        )

        ? meta.visual_results

        : [];


    // Citation source set.
    state.sourceSets.set(
        key,
        sources || []
    );


    // Gallery source set.
    state.sourceSets.set(
        visualKey,
        visualResults
    );


    const row =
        document.createElement(
            "div"
        );

    row.className =
        "message-row assistant-row";


    const visualGallery =
        renderVisualResults(
            visualResults,
            visualKey
        );


    const sourceCards =
        renderSourceCards(
            sources,
            key
        );


    const metaText = [
        meta.confidence
        ? `${
            String(
                meta.confidence
            )
            .charAt(0)
            .toUpperCase()

            + String(
                meta.confidence
            ).slice(1)
        } confidence`
        : "",

        meta.model
        ? friendlyModel(
            meta.model
        )
        : ""
    ]
    .filter(Boolean)
    .join(" · ");


    row.innerHTML = `

        <div class="assistant-avatar">

            <i data-lucide="sparkles"></i>

        </div>


        <div class="assistant-message">

            ${visualGallery}


            <div class="answer-content">

                ${
                    renderMarkdown(
                        markdown,
                        key
                    )
                }

            </div>


            ${sourceCards}


            ${
                metaText

                ? `
                <div class="answer-meta">
                    ${escapeHtml(metaText)}
                </div>
                `

                : ""
            }

        </div>

    `;


    els.messages.appendChild(
        row
    );


    typeset(
        $(".answer-content", row)
    );


    lucide.createIcons();

    return row;
}


function renderSourceCards(
    sources,
    messageKey
) {
    if (
        !sources
        ?.length
    ) {
        return "";
    }

    const cards =
        sources
        .slice(0, 4)
        .map(
            (source, index) => {
                const preview =
                    source.asset_url

                    ? `
                    <img
                        src="${escapeHtml(source.asset_url)}"
                        alt="${escapeHtml(
                            source.label
                            || source.caption
                            || "Retrieved figure"
                        )}"
                        loading="lazy"
                    >
                    `

                    : `
                    <div class="source-card-icon">

                        <i
                            data-lucide="${
                                source.modality
                                === "math"

                                ? "sigma"

                                : source.modality
                                === "table"

                                ? "table-2"

                                : "file-text"
                            }"
                        ></i>

                    </div>
                    `;


                return `

                    <button
                        class="source-card"
                        data-source-set="${escapeHtml(messageKey)}"
                        data-source-index="${index}"
                    >

                        <div class="source-card-preview">
                            ${preview}
                        </div>


                        <div class="source-card-copy">

                            <div class="source-card-kicker">

                                ${
                                    escapeHtml(
                                        source.modality
                                        || "source"
                                    )
                                }

                                · p.

                                ${
                                    escapeHtml(
                                        source.page
                                        || "—"
                                    )
                                }

                            </div>


                            <strong>
                                ${
                                    escapeHtml(
                                        source.title
                                        || "Academic source"
                                    )
                                }
                            </strong>

                        </div>

                    </button>

                `;
            }
        )
        .join("");


    return `

        <div class="sources-block">

            <div class="sources-heading">
                Sources
            </div>

            <div class="source-cards">
                ${cards}
            </div>

        </div>

    `;
}


function appendTyping() {
    els.welcome.classList.add(
        "hidden"
    );

    const row =
        document.createElement(
            "div"
        );

    row.className =
        "message-row assistant-row typing-row";


    row.innerHTML = `

        <div class="assistant-avatar">

            <i data-lucide="sparkles"></i>

        </div>


        <div class="assistant-message">

            <div class="thinking-line">

                <span></span>
                <span></span>
                <span></span>

                <em>
                    Searching the corpus and reasoning…
                </em>

            </div>

        </div>

    `;


    els.messages.appendChild(
        row
    );

    lucide.createIcons();

    return row;
}


function appendError(message) {
    const row =
        document.createElement(
            "div"
        );

    row.className =
        "message-row assistant-row";


    row.innerHTML = `

        <div class="assistant-avatar error">

            <i data-lucide="triangle-alert"></i>

        </div>


        <div class="assistant-message error-message">

            ${escapeHtml(message)}

        </div>

    `;


    els.messages.appendChild(
        row
    );

    lucide.createIcons();
}


function scrollToBottom(
    smooth = true
) {
    requestAnimationFrame(
        () => {
            els.chatScroll.scrollTo(
                {
                    top:
                        els.chatScroll.scrollHeight,

                    behavior:
                        smooth
                        ? "smooth"
                        : "auto"
                }
            );
        }
    );
}


// ============================================================
// Source drawer
// ============================================================

function openSourceDrawer(
    setKey,
    index
) {
    const sources =
        state.sourceSets.get(
            setKey
        )
        || [];

    if (!sources.length) {
        return;
    }

    state.activeSourceSet =
        setKey;

    state.activeSourceIndex =
        Math.max(
            0,

            Math.min(
                Number(index)
                || 0,

                sources.length
                - 1
            )
        );

    renderActiveSource();

    els.drawer.classList.remove(
        "hidden"
    );

    els.appShell.classList.add(
        "drawer-open"
    );
}


function closeSourceDrawer() {
    els.drawer.classList.add(
        "hidden"
    );

    els.appShell.classList.remove(
        "drawer-open"
    );

    state.activeSourceSet =
        null;
}


function renderActiveSource() {
    const sources =
        state.sourceSets.get(
            state.activeSourceSet
        )
        || [];

    const source =
        sources[
            state.activeSourceIndex
        ];

    if (!source) {
        return;
    }

    els.drawerCount.textContent =
        `Source ${
            state.activeSourceIndex
            + 1
        } of ${sources.length}`;

    els.drawerPrev.disabled =
        state.activeSourceIndex
        <= 0;

    els.drawerNext.disabled =
        state.activeSourceIndex
        >= sources.length - 1;


    const visual =
        source.asset_url

        ? `

        <section class="source-section">

            <div class="source-section-label">
                Retrieved visual
            </div>


            <a
                class="source-image-button"
                href="${escapeHtml(source.asset_url)}"
                target="_blank"
                rel="noopener"
            >

                <img
                    class="source-image"
                    src="${escapeHtml(source.asset_url)}"
                    alt="${
                        escapeHtml(
                            source.caption
                            || source.label
                            || "Retrieved figure"
                        )
                    }"
                >

            </a>


            ${
                source.caption

                ? `
                <div class="source-caption">
                    ${escapeHtml(source.caption)}
                </div>
                `

                : ""
            }

        </section>

        `

        : "";


    const equation =
        source.latex

        ? `

        <section class="source-section">

            <div class="source-section-label">
                Equation
            </div>


            <div class="source-equation">

                $$${escapeHtml(
                    stripMathDelimiters(
                        source.latex
                    )
                )}$$

            </div>

        </section>

        `

        : "";


    const table =
        source.modality
        === "table"

        ? renderSourceTable(
            source
        )

        : "";


    const pagePreview =
        source.page_url

        ? `

        <section class="source-section">

            <div class="source-section-row">

                <div class="source-section-label">

                    PDF page
                    ${escapeHtml(source.page || "")}

                </div>


                <a
                    class="open-page-link"
                    href="${escapeHtml(source.page_url)}"
                    target="_blank"
                    rel="noopener"
                >
                    Open page
                </a>

            </div>


            <a
                href="${escapeHtml(source.page_url)}"
                target="_blank"
                rel="noopener"
                class="page-preview-link"
            >

                <img
                    class="page-preview"
                    src="${escapeHtml(source.page_url)}"
                    alt="PDF page ${escapeHtml(source.page || "")}"
                    loading="lazy"
                >

            </a>

        </section>

        `

        : "";


    els.drawerBody.innerHTML = `

        <div class="source-document-head">

            <div class="source-badge">
                ${
                    escapeHtml(
                        source.modality
                        || "source"
                    )
                }
            </div>


            <h2>

                ${
                    escapeHtml(
                        source.title
                        || "Academic source"
                    )
                }

            </h2>


            <div class="source-meta">

                ${
                    source.author

                    ? `
                    <span>
                        ${escapeHtml(source.author)}
                    </span>
                    `

                    : ""
                }


                ${
                    source.category

                    ? `
                    <span>
                        ${escapeHtml(source.category)}
                    </span>
                    `

                    : ""
                }


                ${
                    source.page

                    ? `
                    <span>
                        Page ${escapeHtml(source.page)}
                    </span>
                    `

                    : ""
                }

            </div>


            ${
                source.section

                ? `
                <div class="source-section-path">

                    ${escapeHtml(source.section)}

                </div>
                `

                : ""
            }

        </div>


        ${visual}

        ${equation}

        ${table}


        ${
            source.excerpt

            ? `

            <section class="source-section">

                <div class="source-section-label">
                    Retrieved evidence
                </div>


                <div class="source-excerpt">

                    ${
                        escapeHtml(
                            source.excerpt
                        )
                        .replace(
                            /\n/g,
                            "<br>"
                        )
                    }

                </div>

            </section>

            `

            : ""
        }


        ${pagePreview}

    `;


    typeset(
        els.drawerBody
    );

    lucide.createIcons();
}


function stripMathDelimiters(
    value
) {
    let text =
        String(
            value || ""
        ).trim();

    text = text.replace(
        /^\$\$|\$\$$/g,
        ""
    );

    text = text.replace(
        /^\\\[|\\\]$/g,
        ""
    );

    text = text.replace(
        /^\\\(|\\\)$/g,
        ""
    );

    return text.trim();
}


function renderSourceTable(
    source
) {
    const cells =
        Array.isArray(
            source.cells
        )

        ? source.cells

        : [];

    if (!cells.length) {
        const rows =
            (
                source.rows
                || []
            )
            .map(
                row => `

                <li>
                    ${
                        escapeHtml(
                            row.text
                            || String(row)
                        )
                    }
                </li>

                `
            )
            .join("");

        return rows

        ? `

        <section class="source-section">

            <div class="source-section-label">
                Retrieved table rows
            </div>


            <ul class="source-row-list">

                ${rows}

            </ul>

        </section>

        `

        : "";
    }


    const rowIds = [
        ...new Set(
            cells.map(
                cell =>
                    Number(
                        cell.row
                    )
            )
        )
    ].sort(
        (a, b) =>
            a - b
    );


    const colIds = [
        ...new Set(
            cells.map(
                cell =>
                    Number(
                        cell.col
                    )
            )
        )
    ].sort(
        (a, b) =>
            a - b
    );


    const cellMap =
        new Map(
            cells.map(
                cell => [
                    `${
                        Number(
                            cell.row
                        )
                    }:${
                        Number(
                            cell.col
                        )
                    }`,

                    cell
                ]
            )
        );


    const body =
        rowIds.map(
            row => `

            <tr>

                ${
                    colIds.map(
                        col => {
                            const cell =
                                cellMap.get(
                                    `${row}:${col}`
                                );

                            return `

                            <td>
                                ${
                                    escapeHtml(
                                        cell?.raw_value
                                        ?? ""
                                    )
                                }
                            </td>

                            `;
                        }
                    ).join("")
                }

            </tr>

            `
        ).join("");


    return `

        <section class="source-section">

            <div class="source-section-label">
                Retrieved table cells
            </div>


            <div class="table-scroll">

                <table class="source-table">

                    <tbody>
                        ${body}
                    </tbody>

                </table>

            </div>

        </section>

    `;
}


// ============================================================
// Composer
// ============================================================

function autoResize() {
    els.input.style.height =
        "auto";

    els.input.style.height =
        `${
            Math.min(
                els.input.scrollHeight,
                180
            )
        }px`;
}


async function submitQuestion(
    event
) {
    event?.preventDefault();

    if (state.busy) {
        return;
    }

    const query =
        els.input.value.trim();

    if (!query) {
        return;
    }

    state.busy =
        true;

    els.send.disabled =
        true;

    appendUserMessage(
        query
    );

    els.input.value =
        "";

    autoResize();

    scrollToBottom();

    const typing =
        appendTyping();

    scrollToBottom();

    try {
        const response =
            await api(
                "/api/chat",

                {
                    method:
                        "POST",

                    body:
                        JSON.stringify(
                            {
                                query,

                                conversation_id:
                                    state.conversationId,

                                model:
                                    state.prefs.model,

                                style:
                                    state.prefs.answerStyle,

                                reasoning:
                                    state.prefs.reasoning
                            }
                        )
                }
            );

        const body =
            await response.json();

        state.conversationId =
            body.conversation_id
            || state.conversationId;

        typing.remove();

        appendAssistantMessage(
            body.answer
            || "",

            body.sources
            || [],

            {
                confidence:
                    body.confidence,

                model:
                    body.model,

                status:
                    body.status,

                visual_results:
                    body.visual_results
                    || []
            }
        );

        await loadConversations();

    } catch (error) {
        typing.remove();

        appendError(
            error.message
            || "Something went wrong."
        );

    } finally {
        state.busy =
            false;

        els.send.disabled =
            false;

        els.input.focus();

        scrollToBottom();
    }
}


// ============================================================
// Model + settings
// ============================================================

function positionModelMenu() {
    const rect =
        els.modelCard.getBoundingClientRect();

    els.modelMenu.style.left =
        `${Math.max(
            12,
            rect.left
        )}px`;

    els.modelMenu.style.bottom =
        `${
            Math.max(
                12,

                window.innerHeight
                - rect.top
                + 8
            )
        }px`;
}


function showModelMenu() {
    positionModelMenu();

    els.modelMenu.classList.remove(
        "hidden"
    );
}


function hideModelMenu() {
    els.modelMenu.classList.add(
        "hidden"
    );
}


function openSettings() {
    syncPrefsUI();

    els.settingsModal.classList.remove(
        "hidden"
    );

    requestAnimationFrame(
        () =>
            els.settingsModal.classList.add(
                "open"
            )
    );
}


function closeSettings() {
    els.settingsModal.classList.remove(
        "open"
    );

    setTimeout(
        () =>
            els.settingsModal.classList.add(
                "hidden"
            ),

        160
    );
}


function saveSettingsFromModal() {
    state.prefs.name =
        els.profileName.value.trim()
        || DEFAULT_PREFS.name;

    state.prefs.role =
        els.profileRole.value.trim()
        || DEFAULT_PREFS.role;

    state.prefs.model =
        els.settingsModel.value
        || state.prefs.model;

    state.prefs.answerStyle =
        els.answerStyle.value;

    state.prefs.reasoning =
        els.reasoningMode.value;

    savePrefs();

    renderModels();

    closeSettings();
}


// ============================================================
// Events
// ============================================================

els.form.addEventListener(
    "submit",
    submitQuestion
);


els.input.addEventListener(
    "input",
    autoResize
);


els.input.addEventListener(
    "keydown",

    event => {
        if (
            event.key
            === "Enter"

            && !event.shiftKey
        ) {
            event.preventDefault();

            submitQuestion(
                event
            );
        }
    }
);


els.newChat.addEventListener(
    "click",
    newChat
);


els.search.addEventListener(
    "input",
    renderHistory
);


els.modelCard.addEventListener(
    "click",

    event => {
        event.stopPropagation();

        if (
            els.modelMenu.classList.contains(
                "hidden"
            )
        ) {
            showModelMenu();

        } else {
            hideModelMenu();
        }
    }
);


els.settingsCard.addEventListener(
    "click",
    openSettings
);


els.settingsClose.addEventListener(
    "click",
    closeSettings
);


els.settingsCancel.addEventListener(
    "click",
    closeSettings
);


els.settingsSave.addEventListener(
    "click",
    saveSettingsFromModal
);


els.drawerClose.addEventListener(
    "click",
    closeSourceDrawer
);


els.drawerPrev.addEventListener(
    "click",

    () => {
        state.activeSourceIndex -= 1;

        renderActiveSource();
    }
);


els.drawerNext.addEventListener(
    "click",

    () => {
        state.activeSourceIndex += 1;

        renderActiveSource();
    }
);


els.settingsModal.addEventListener(
    "click",

    event => {
        if (
            event.target
            === els.settingsModal
        ) {
            closeSettings();
        }
    }
);


document.addEventListener(
    "click",

    event => {
        const sourceButton =
            event.target.closest(
                "[data-source-set][data-source-index]"
            );

        if (sourceButton) {
            openSourceDrawer(
                sourceButton.dataset.sourceSet,

                Number(
                    sourceButton.dataset.sourceIndex
                )
            );

            return;
        }

        if (
            !event.target.closest(
                "#modelMenu"
            )

            && !event.target.closest(
                "#modelCard"
            )
        ) {
            hideModelMenu();
        }
    }
);


window.addEventListener(
    "resize",

    () => {
        if (
            !els.modelMenu.classList.contains(
                "hidden"
            )
        ) {
            positionModelMenu();
        }
    }
);


$$("[data-suggestion]")
    .forEach(
        button => {
            button.addEventListener(
                "click",

                () => {
                    els.input.value =
                        button.dataset.suggestion;

                    autoResize();

                    els.input.focus();
                }
            );
        }
    );


// ============================================================
// Boot
// ============================================================

async function boot() {
    syncPrefsUI();

    autoResize();

    lucide.createIcons();

    await Promise.all(
        [
            loadHealth(),
            loadModels(),
            loadConversations()
        ]
    );

    lucide.createIcons();

    els.input.focus();
}


boot();