$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    
	$('#imageUpload').change(
                function () {
                    var fileExtension = ['jpeg', 'jpg'];
                    if ($.inArray($(this).val().split('.').pop().toLowerCase(), fileExtension) == -1) {
                        alert("Only '.jpeg','.jpg','.pdf' formats are allowed.");
                        return false; }
});

    