var app = angular.module('app', ['tableSort']);

app
  .controller('MainController', function MainController($scope, $http) {
    $scope.models = [];

    $http({
      method: 'GET',
      url: 'models.yaml'
    }).then(function successCallback(response) {
      $scope.models = jsyaml.load(response.data).models; // response data
    }, function errorCallback(error) {
      console.error(error);
    });
  })
  .filter('parseCurrency', function () {
    return function (input) {
      return input.replace('$', '').replace(/,/g, '');
    };
  });
