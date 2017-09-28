var app = angular.module('app', ['tableSort', 'pathgather.popeye', 'ngFileUpload']);

app
  .controller('MainController', function MainController($scope, $http, Popeye, Upload) {
    $scope.models = [];
    $scope.form = {};
    $scope.icons = {'Computer Vision': 'computer_vision.png',
      'Natural Language Processing': 'nlp.png'};
    $http({
      method: 'GET',
      url: 'models.yaml'
    }).then(function successCallback(response) {
      $scope.models = jsyaml.load(response.data).models; // response data
    }, function errorCallback(error) {
      console.error(error);
    });

    $scope.openModal = function () {
      Popeye.openModal({
        templateUrl: "demoModal.html",
        scope: $scope,
        modalClass: 'demo-modal'
      });
    };
    $scope.openModal();

    $scope.uploadImage = function (file) {
      Upload.upload({
        url: 'http://s2.ndres.me:8091/vgg16',
        file: file,
      }).progress(function(e) {
      }).then(function(data, status, headers, config) {
        $scope.predictions = data.data
      });
    }
  })
  .filter('parseCurrency', function () {
    return function (input) {
      return input.replace('$', '').replace(/,/g, '');
    };
  });
