// ===============================
// Guillermo Ramajo Portfolio
// main.js
// ===============================

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------
    // Mobile menu
    // -----------------------------
    const menuBtn = document.getElementById("menu-btn");
    const navMenu = document.getElementById("nav-menu");

    if (menuBtn && navMenu) {
        menuBtn.addEventListener("click", () => {
            navMenu.classList.toggle("show");
        });
    }

    // -----------------------------
    // Highlight current page
    // -----------------------------
    const current = location.pathname.split("/").pop() || "index.html";

    document.querySelectorAll("nav a").forEach(link => {
        if (link.getAttribute("href") === current) {
            link.classList.add("active");
        }
    });

    // -----------------------------
    // Back to top
    // -----------------------------
    const topBtn = document.getElementById("top-btn");

    if (topBtn) {

        window.addEventListener("scroll", () => {
            topBtn.classList.toggle("visible", window.scrollY > 500);
        });

        topBtn.addEventListener("click", () => {
            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });
        });

    }

    // -----------------------------
    // Footer year
    // -----------------------------
    const year = document.getElementById("year");

    if (year) {
        year.textContent = new Date().getFullYear();
    }

});