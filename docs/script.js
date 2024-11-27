get = id => document.getElementById(id);

function author_node(author) {
    var span = document.createElement("span");
    var a = document.createElement("a");
    var sup = document.createElement("sup");
    a.textContent = author.name;
    sup.textContent = author.footnote.map(String).join(",");
    sup.textContent += author.affiliations.map(String).join(",");
    span.appendChild(a);
    span.appendChild(sup);
    return span
}

function affiliations_node(affiliations) {
    var span = document.createElement("span");
    span.innerHTML = affiliations.map((affiliation, index) =>
        "<sup>" + (index + 1).toString() + "</sup>" + affiliation
    ).join(", ");
    return span
}

function footnote_node(footnotes) {
    var span = document.createElement("span");
    // footnotes is a list of pairs of the form [symbol, footnote]
    // Then make a string of the form "<sup>symbol</sup> footnote"
    // Then join the strings with ", "
    span.innerHTML = footnotes.map(footnote =>
        "<sup>" + footnote[0] + "</sup>" + footnote[1]
    ).join(", ");
    return span
}

function make_site(paper) {
    document.title = paper.title;
    get("title").textContent = paper.title;
    get("conference").textContent = paper.conference;

    
    paper.authors.map((author, index) => {
        node = author_node(author);
        get("author-list").appendChild(node);
        if (index == paper.authors.length - 1) return;
        node.innerHTML += ", "
    })
    get("affiliation-list").appendChild(affiliations_node(paper.affiliations));
    //get("footnote-list").appendChild(footnote_node(paper.footnotes));
    get("abstract").textContent = paper.abstract;

    // Add the video under the abstract if it exists
    if (paper.video) {
        var videoDiv = get("video");
        var video = document.createElement("video");
        video.width = 640;
        video.height = 640;
        video.controls = true;

        var source = document.createElement("source");
        source.src = paper.video;
        source.type = "video/mp4";

        video.appendChild(source);
        videoDiv.appendChild(video);
    }

    // Populate the button list with the URLs from the paper
    buttonlist = get("button-list");
    for (var button in paper.URLs) {
        node = document.createElement("a");
        node.href = paper.URLs[button];

        img = document.createElement("img");
        img.src = "assets/logos/arXiv.svg";
        node.appendChild(img);

        span = document.createElement("span");
        span.textContent = button;
        node.appendChild(span);

        buttonlist.appendChild(node);
    }

    // Create the citation node at the end of the page in the bibtex div
    // and add a copy button to the bibtex div
    // bibtex = get("bibtex");
    // bibtextext = document.createElement("div");
    // bibtextext.id = "bibtex-text";
    // bibtextext.textContent = atob(paper.base64bibtex);
    // var button = document.createElement("button");
    // button.id = "copy-button";
    // button.textContent = "Copy";
    // button.onclick = () => {
    //     var range = document.createRange();
    //     range.selectNode(bibtextext);
    //     window.getSelection().removeAllRanges();
    //     window.getSelection().addRange(range);
    //     document.execCommand("copy");
    //     window.getSelection().removeAllRanges();
    // }
    // bibtex.appendChild(button);
    // bibtex.appendChild(bibtextext);
}

function set_slider(root) {
    const slidesContainer = root.querySelector(".slides-container");
    const slide = root.querySelector(".slide");
    const prevButton = root.querySelector(".slide-arrow-prev");
    const nextButton = root.querySelector(".slide-arrow-next");
    nextButton.addEventListener("click", (event) => {
        const slideWidth = slide.clientWidth;
        slidesContainer.scrollLeft += slideWidth;
    });
    prevButton.addEventListener("click", () => {
        const slideWidth = slide.clientWidth;
        slidesContainer.scrollLeft -= slideWidth;
    });
}

fetch("./paper.json").then(response => response.json()).then(json => make_site(json));

sliders = document.getElementsByClassName("slider-wrapper")
for (var i = 0; i < sliders.length; i++) set_slider(sliders[i])

