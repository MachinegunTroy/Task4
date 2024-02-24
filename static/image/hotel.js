document.addEventListener("click", function(event) {
    if (!event.target.matches('.amenities-toggle-label')) {
        var dropdowns = document.querySelectorAll('.amenities-dropdown.open');
        dropdowns.forEach(function(dropdown) {
            dropdown.classList.remove('open');
        });
    }
});

document.querySelector('.amenities-toggle-label').addEventListener('click', function() {
    var dropdown = this.nextElementSibling;
    dropdown.classList.toggle('open');
});