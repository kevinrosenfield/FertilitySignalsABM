function set_up_model() {

    // Get the correlation value
    var rankFitnessCorrelation = $("#rankFitnessCorrelation").val();
    var nMales = $("#nMales").val();
    var nFemales = $("#nFemales").val();
    var nGroups = $("#nGroups").val();
    var nGenerations = $("#nGenerations").val();

    // Create a new FormData object
    var formData = new FormData();
    formData.append('rankFitnessCorrelation', rankFitnessCorrelation);
    formData.append('nMales', nMales);
    formData.append('nFemales', nFemales);
    formData.append('nGroups', nGroups);
    formData.append('nGenerations', nGenerations);
    var xhr = new XMLHttpRequest();

    // Set up the request
    xhr.open('POST', '/set-up-model', true);

    // Send the request without a payload

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            // Perform any additional actions after the form is submitted

            plot1();
            plot2();
            info();

        }
            
    };

    xhr.send(formData);
    stop = false

    // enabled action buttons
    document.getElementById("one-generation-button").disabled = false; // Make the button inactive

    document.getElementById("one-day-button").disabled = false; // Make the button inactive

    document.getElementById("evolve-button").disabled = false; // Make the button inactive

    document.getElementById("stop-button").disabled = true; // Make the button inactive

};

function plot1() {

    // plot2()
    // Make a request to the server to generate the plot
    fetch('/image-endpoint', {
        method: 'POST',
    })
    .then(response => response.text())
    .then(data => {
        // Update the plot image sources
        $("#rank-fitness-container img").attr("src", 'data:image/png;base64,' + data);
    
        // Show the plot containers
        document.getElementById("plot-container").style.display = "grid";
    })
    .catch(error => {
        console.error('Error fetching image:', error);
    });
};

function plot2() {

    // Make a request to the server to generate the plot
    fetch('/image-endpoint2', {
        method: 'POST',
    })
    .then(response => response.text())
    .then(data => {
    
        // Update the plot image sources
        $("#rank-RS-container img").attr("src", 'data:image/png;base64,' + data);
    
        // Show the plot containers
        document.getElementById("plot-container").style.display = "grid";
    })
    .catch(error => {
        console.error('Error fetching image:', error);
    });
};

function info() {
// Make an AJAX request to the server to retrieve data
    $.ajax({
        url: '/data',
        method: 'GET',
        success: function(response) {
    
        // Use the data in your JavaScript functions
        document.getElementById("info-container").style.display = "block";
        document.getElementById("info").innerHTML = response;

        }
    });
};

function go_one_generation() {
    // Prevent the default form submission behavior

    document.getElementById("stop-button").disabled = false; // Make the button inactive

    var xhr = new XMLHttpRequest();
    
    // Set up the request
    xhr.open('POST', '/go-one-generation', true);

    xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
        // Perform any additional actions after the form is submitted

        plot1();
        plot2();
        info();
    }
};

// Send the request without a payload
xhr.send();

    // You can optionally disable the button to prevent multiple submissions
    // $(this).find('input[type="submit"]').prop('disabled', true);

    // document.getElementById("stop-button").disabled = true; // Make the button inactive

};

function go_one_day() {
    // Prevent the default form submission behavior

    document.getElementById("stop-button").disabled = false; // Make the button inactive

    var xhr = new XMLHttpRequest();
    
    // Set up the request
    xhr.open('POST', '/go-one-day', true);

    xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
        // Perform any additional actions after the form is submitted

        plot1();
        plot2();
        info();
    }
};

// Send the request without a payload
xhr.send();

    // You can optionally disable the button to prevent multiple submissions
    // $(this).find('input[type="submit"]').prop('disabled', true);

    // document.getElementById("stop-button").disabled = true; // Make the button inactive

};

function evolve() {
        
    document.getElementById("stop-button").disabled = false; // Make the button inactive

    if (stop === false) {
        var sendRequest = function() {
            console.log(stop_loop)
            if (stop_loop === true) { stop_loop = false; return ''}
            // get model speed
            var speed = $("#speed").val();

            // Create a new FormData object
            var formData = new FormData();
            formData.append('speed', speed);

            var xhr = new XMLHttpRequest();
            
            // Set up the request
            xhr.open('POST', '/evolve', true);
            
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    var response = xhr.responseText;
                    console.log(response.trim())
                    if (response.trim() != 'true') {
                        // Response is not true, perform additional actions
                        
                        
                        plot1();
                        plot2();
                        info();

                        // Send the next request
                        sendRequest();
                    } else {
                        stop = true
                    }
                }
            };
            
            // Send the request without a payload
            xhr.send(formData);
        };
        // Start the process by sending the first request
        sendRequest();
    };
    
    // document.getElementById("stop-button").disabled = true; // Make the button inactive
};

var stop_loop = false

function stopLoop() {
    stop_loop = true;
    document.getElementById("stop-button").disabled = true
}

function executeCommand() {
var command = document.getElementById("commandInput").value;

// Make a request to the Flask server
fetch('/execute-command', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ command: command })
})

.then(response => response.text())
.then(data => {
    try {
        // Handle the response data
        document.getElementById("terminal-output").innerHTML = data;
        console.log(data);
    } catch (error) {
        console.error('Error parsing JSON:', error);
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
  }

  function inc(name) {
    let number = document.querySelector('[name="' + name + '"]');
    number.value = parseInt(number.value) + 1;
  }
  
  function dec(name) {
    let number = document.querySelector('[name="' + name + '"]');
      if (parseInt(number.value) > 0) {
        number.value = parseInt(number.value) - 1;
    }
  }

  function inc_decimal(name) {
    let number = document.querySelector('[name="' + name + '"]');
    number.value = parseFloat(number.value) + 0.01;
    number.value = Math.round(number.value * 100) / 100
  }
  
  function dec_decimal(name) {
    let number = document.querySelector('[name="' + name + '"]');
      if (parseFloat(number.value) > 0) {
        number.value = parseFloat(number.value) - 0.01;
        number.value = Math.round(number.value * 100) / 100
    }
  }

$(document).ready(function () {

    // disable all action buttons other than setup

    document.getElementById("setup-button").disabled = false; // Make the button inactive

    document.getElementById("one-generation-button").disabled = true; // Make the button inactive

    document.getElementById("one-day-button").disabled = true; // Make the button inactive

    document.getElementById("evolve-button").disabled = true; // Make the button inactive

    document.getElementById("stop-button").disabled = true; // Make the button inactive

    document.getElementById('stop-button').addEventListener('click', stopLoop);
    
    // document.getElementById('setup-button').addEventListener('click',document.getElementById("stop-button").disabled = true); // Make the button inactive);

    document.getElementById('hide-show-button').addEventListener('click',hide_show); // Make the button inactive);

    function hide_show() {

        // Get all elements with the class "my-class"
        const elements = document.querySelectorAll('.variable');
        
        if (document.getElementById("variable-input-container").style.height === "0px")
        {

            document.getElementById("variable-input-container").style.height = "auto";
            document.getElementById("variable-input-container").style.padding = '8px';
        
            // Loop through each element and modify the style
            elements.forEach(element => {
                element.style.display = 'flex';
            });
         } else {
            
            document.getElementById("variable-input-container").style.height = 0;
            document.getElementById("variable-input-container").style.padding = '3px';
        
            // Loop through each element and modify the style
            elements.forEach(element => {
                element.style.display = 'none';
        });

        }
        
      }

    const input = document.getElementById('commandInput');
    const button = document.getElementById('executeButton');

    input.addEventListener('keydown', function(event) {
        if (event.key === "Enter") {
            if (event.shiftKey) {
                return '' } else {
            event.preventDefault();
            button.click(); // Trigger the button's click event
        }
    }
    });

    document.getElementById("executeButton").addEventListener("click", executeCommand);

    const textarea = document.getElementById('commandInput');
  
    textarea.addEventListener('input', () => {
    textarea.style.height = 'auto'; // Reset the height to auto
    textarea.style.height = textarea.scrollHeight + 'px'; // Set the height to match the content
  });
});