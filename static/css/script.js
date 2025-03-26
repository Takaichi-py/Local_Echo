document.addEventListener("DOMContentLoaded", function () {
    const buttons = document.querySelectorAll(".bottom-nav button, .nav-btn, .recommendation-item");

    buttons.forEach(button => {
        button.addEventListener("click", function () {
            const section = this.getAttribute("onclick").replace("navigateTo('", "").replace("')", "");
            navigateTo(section);
        });
    });
});

function navigateTo(section) {
    alert("ページ遷移: " + section);
}

function translateText() {
    const text = document.querySelector("#input-text").value.trim();
    const dialect = document.querySelector("#dialect-select").value;

    if (text === "") {
        alert("翻訳する文章を入力してください。");
        return;
    }

    let translatedText = (dialect === "kansai") ? text + "やで！" : text + "ばい！";
    alert("翻訳結果: " + translatedText);
}


